#!/usr/bin/env python
"""Build a homology-controlled train/valid/test split for nanobody-antigen complexes.

Why this exists
---------------
The shipped splits were produced by ``random.shuffle`` over the record list with no
clustering and no random seed (see MEAN/split_nano_hern.py). Because SabDab contains
many near-duplicate entries (the same nanobody or the same complex solved repeatedly),
a random split leaks: in ``abgnn/finetune/exp2-hern/nanobody`` 84.5% of test CDR3s
occur verbatim in training, 64.7% of test PDB ids also appear in training, and a
trivial "copy the CDR3 of the nearest training framework" baseline scores 79.5% AAR --
higher than any published model on that benchmark. Such a split cannot rank methods.

What this does
--------------
1. Groups records with a union-find over two relations:
     * identical PDB id                    (never split one entry across sets)
     * same MMseqs2 CDR3 cluster           (--min-seq-id / --cov, below)
2. Assigns whole groups to train/valid/test, greedily filling whichever split is
   furthest below its target fraction, with a fixed seed so the split is reproducible.
3. Reports residual leakage, which should be exactly zero on all three measures.

Usage
-----
    python tools/make_clustered_split.py \
        --input-dir  abgnn/finetune/exp2-hern/nanobody \
        --output-dir abgnn/finetune/exp2-hern/nanobody_clustered \
        --min-seq-id 0.6 --cov 0.8 --ratios 0.8 0.1 0.1 --seed 42

``mmseqs`` must be on PATH (it ships in the ``ABGNN`` conda env).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile


def cdr3_of(entry):
    return ''.join(a for a, t in zip(entry['antibody_seq'], entry['antibody_cdr']) if t == '3')


def load_records(input_dir):
    records = []
    for split in ('train', 'valid', 'test'):
        path = os.path.join(input_dir, f'{split}_data.jsonl')
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            for line in handle:
                records.append(json.loads(line))
    if not records:
        sys.exit(f'no *_data.jsonl found under {input_dir}')
    return records


def sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def cluster_cdr3(sequences, min_seq_id, cov, workdir, threads):
    """Return {cdr3 -> cluster representative} using MMseqs2."""
    if shutil.which('mmseqs') is None:
        sys.exit('mmseqs not found on PATH (it is installed in the ABGNN conda env)')
    fasta = os.path.join(workdir, 'cdr3.fasta')
    tag = {f'c{i}': s for i, s in enumerate(sequences)}
    with open(fasta, 'w') as handle:
        for name, seq in tag.items():
            handle.write(f'>{name}\n{seq}\n')
    subprocess.run(
        ['mmseqs', 'easy-cluster', fasta, os.path.join(workdir, 'clu'),
         os.path.join(workdir, 'tmp'), '--min-seq-id', str(min_seq_id),
         '-c', str(cov), '--cov-mode', '1', '-s', '7.5',
         '--threads', str(threads)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    representative = {}
    with open(os.path.join(workdir, 'clu_cluster.tsv')) as handle:
        for line in handle:
            rep, member = line.split()
            representative[tag[member]] = tag[rep]
    return representative


def build_groups(records, representative, antigen_representative=None):
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, entry in enumerate(records):
        union(('entry', i), ('pdb', entry['pdb']))
        union(('entry', i), ('cdr3', representative[cdr3_of(entry)]))
        if antigen_representative:
            antigen = entry.get('antigen_seq', '')
            if antigen:
                union(('entry', i), ('antigen', antigen_representative[antigen]))

    groups = collections.defaultdict(list)
    for i in range(len(records)):
        groups[find(('entry', i))].append(i)
    return list(groups.values())


def assign(groups, total, ratios, seed):
    names = ('train', 'valid', 'test')
    targets = dict(zip(names, ratios))
    groups = list(groups)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)          # place big clusters first
    counts = {n: 0 for n in names}
    placement = {}
    for group in groups:
        pick = max(names, key=lambda n: targets[n] * total - counts[n])
        for i in group:
            placement[i] = pick
        counts[pick] += len(group)
    return placement, counts


def report_leakage(records, representative, placement, antigen_representative=None):
    indices = {
        split: [i for i, assigned in placement.items() if assigned == split]
        for split in ('train', 'valid', 'test')
    }
    metrics = {
        'identical_cdr3': lambda entry: cdr3_of(entry),
        'shared_pdb_id': lambda entry: entry['pdb'],
        'shared_cdr3_cluster': lambda entry: representative[cdr3_of(entry)],
    }
    if antigen_representative is not None:
        metrics.update(
            {
                'identical_antigen': lambda entry: entry.get('antigen_seq', ''),
                'shared_antigen_cluster': lambda entry: antigen_representative.get(
                    entry.get('antigen_seq', ''), ''
                ),
            }
        )

    report = {}
    for left, right in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        comparison = f'{right}_vs_{left}'
        report[comparison] = {}
        print(f'  {right}-vs-{left} leakage over {len(indices[right])} {right} records:')
        for name, value_fn in metrics.items():
            left_values = {
                value_fn(records[index]) for index in indices[left]
                if value_fn(records[index])
            }
            count = sum(
                bool(value_fn(records[index]))
                and value_fn(records[index]) in left_values
                for index in indices[right]
            )
            report[comparison][name] = count
            print(f'    {name.replace("_", " "):23s}: {count}')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--min-seq-id', type=float, default=0.6,
                    help='MMseqs2 CDR3 identity threshold (default 0.6)')
    ap.add_argument('--cov', type=float, default=0.8, help='MMseqs2 coverage (default 0.8)')
    ap.add_argument('--ratios', type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    metavar=('TRAIN', 'VALID', 'TEST'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--antigen-disjoint', action='store_true',
                    help='additionally forbid an antigen cluster from spanning splits, so '
                         'the test set contains only antigens unseen in training '
                         '(answers "generalisation to entirely novel antigens")')
    ap.add_argument('--antigen-min-seq-id', type=float, default=0.4)
    ap.add_argument('--antigen-cov', type=float, default=0.6)
    args = ap.parse_args()
    if args.threads <= 0:
        ap.error('--threads must be positive')

    records = load_records(args.input_dir)
    unique_cdr3 = sorted({cdr3_of(e) for e in records})
    print(f'{len(records)} records, {len(unique_cdr3)} unique CDR3s')

    workdir = tempfile.mkdtemp(prefix='clustersplit_')
    try:
        representative = cluster_cdr3(
            unique_cdr3, args.min_seq_id, args.cov, workdir, args.threads
        )
        antigen_representative = None
        if args.antigen_disjoint:
            antigens = sorted({e['antigen_seq'] for e in records if e.get('antigen_seq')})
            sub = os.path.join(workdir, 'ag')
            os.makedirs(sub, exist_ok=True)
            antigen_representative = cluster_cdr3(
                antigens, args.antigen_min_seq_id, args.antigen_cov, sub, args.threads
            )
            print(f'{len(set(antigen_representative.values()))} antigen clusters '
                  f'(min-seq-id={args.antigen_min_seq_id}, cov={args.antigen_cov})')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f'{len(set(representative.values()))} CDR3 clusters '
          f'(min-seq-id={args.min_seq_id}, cov={args.cov})')

    groups = build_groups(records, representative, antigen_representative)
    print(f'{len(groups)} connected groups after union(PDB, CDR3-cluster)')

    placement, counts = assign(groups, len(records), args.ratios, args.seed)
    print(f'split sizes: {counts}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = {}
    for split in ('train', 'valid', 'test'):
        out = os.path.join(args.output_dir, f'{split}_data.jsonl')
        output_paths[split] = out
        with open(out, 'w') as handle:
            for i, entry in enumerate(records):
                if placement[i] == split:
                    handle.write(json.dumps(entry) + '\n')
    print(f'wrote {args.output_dir}/{{train,valid,test}}_data.jsonl')

    leakage = report_leakage(
        records, representative, placement, antigen_representative
    )
    if any(
        count
        for comparison in leakage.values()
        for count in comparison.values()
    ):
        sys.exit('ERROR: residual leakage is non-zero')
    print('  -> zero pairwise leakage for every audited relation')

    source_paths = {
        split: os.path.join(args.input_dir, f'{split}_data.jsonl')
        for split in ('train', 'valid', 'test')
        if os.path.exists(os.path.join(args.input_dir, f'{split}_data.jsonl'))
    }
    metadata = {
        'input_dir': os.path.abspath(args.input_dir),
        'record_count': len(records),
        'unique_cdr3_count': len(unique_cdr3),
        'cdr3_cluster_count': len(set(representative.values())),
        'antigen_cluster_count': (
            len(set(antigen_representative.values()))
            if antigen_representative is not None
            else None
        ),
        'connected_group_count': len(groups),
        'parameters': {
            'min_seq_id': args.min_seq_id,
            'coverage': args.cov,
            'ratios': args.ratios,
            'seed': args.seed,
            'antigen_disjoint': args.antigen_disjoint,
            'antigen_min_seq_id': args.antigen_min_seq_id,
            'antigen_coverage': args.antigen_cov,
        },
        'split_counts': counts,
        'pairwise_leakage_record_counts': leakage,
        'source_sha256': {split: sha256(path) for split, path in source_paths.items()},
        'output_sha256': {split: sha256(path) for split, path in output_paths.items()},
    }
    metadata_path = os.path.join(args.output_dir, 'split_metadata.json')
    with open(metadata_path, 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f'wrote {metadata_path}')


if __name__ == '__main__':
    main()
