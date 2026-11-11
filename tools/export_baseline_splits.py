#!/usr/bin/env python
"""Export a clustered split into the formats the Table 1 baselines consume.

All baselines must be re-trained and re-evaluated on the *same* homology-clustered split,
otherwise their numbers are not comparable to each other (see docs/REPRODUCIBILITY.md, Data).
The split is defined so that a PDB entry never spans two sets, which makes a plain
PDB-id -> split mapping well defined and usable by any tool that indexes by PDB.

Emits, under --output-dir:

  pdb_assignment.csv        pdb,split               universal; use to filter any pipeline
  refinegnn/{train,valid,test}.jsonl
                            RefineGNN / structgen native: {seq, cdr, coords{N,CA,C,O}}
  mean/{train,valid,test}.json
                            MEAN native: entries of MEAN/summaries/sabdab_all.json
                            restricted to the split, with pdb_data_path rewritten to
                            wherever the .pdb files actually live now
  coverage.txt              per-split coverage + the PDB ids each baseline cannot cover

Example
-------
    python tools/export_baseline_splits.py \
        --split-dir  abgnn/finetune/exp2-hern/nanobody_clustered \
        --output-dir baselines/nanobody_clustered \
        --mean-summary MEAN/summaries/sabdab_all.json \
        --pdb-dirs MEAN/summaries/pdb MEAN/summaries/all_nano_structures
"""
import argparse
import json
import os
import sys

SPLITS = ('train', 'valid', 'test')
BACKBONE = ('N', 'CA', 'C', 'O')      # atom14 slots 0..3


def load_split(split_dir):
    """{split: [record]} plus {pdb: split}; fails loudly if a PDB spans splits."""
    records, assignment = {}, {}
    for split in SPLITS:
        path = os.path.join(split_dir, f'{split}_data.jsonl')
        records[split] = []
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            for line in handle:
                entry = json.loads(line)
                records[split].append(entry)
                pdb = entry['pdb'].lower()
                if assignment.setdefault(pdb, split) != split:
                    sys.exit(f'PDB {pdb} spans splits; the split is not PDB-level and '
                             f'baselines indexed by PDB cannot use it')
    return records, assignment


def write_refinegnn(records, out_dir):
    """RefineGNN reads {seq, cdr, coords{atom: [L,3]}} — structgen/data.py."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split in SPLITS:
        path = os.path.join(out_dir, f'{split}.jsonl')
        with open(path, 'w') as handle:
            for entry in records[split]:
                coords = entry['antibody_coords']          # [L][14][3]
                handle.write(json.dumps({
                    'pdb': entry['pdb'],
                    'seq': entry['antibody_seq'],
                    'cdr': entry['antibody_cdr'],
                    'coords': {atom: [residue[i] for residue in coords]
                               for i, atom in enumerate(BACKBONE)},
                }) + '\n')
        counts[split] = len(records[split])
    return counts


def load_mean_summary(path):
    with open(path) as handle:
        text = handle.read().lstrip()
    handle = open(path)
    entries = json.load(handle) if text.startswith('[') else [json.loads(l) for l in handle]
    handle.close()
    return entries


def write_mean(entries, assignment, pdb_dirs, out_dir):
    """MEAN reads its own summary json and opens pdb_data_path — MEAN/data/dataset.py."""
    located = {}
    for directory in pdb_dirs:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.endswith('.pdb'):
                located.setdefault(name[:-4].lower(), os.path.abspath(
                    os.path.join(directory, name)))

    os.makedirs(out_dir, exist_ok=True)
    buckets = {s: [] for s in SPLITS}
    missing_pdb_file = set()
    for entry in entries:
        pdb = entry['pdb'].lower()
        split = assignment.get(pdb)
        if split is None:
            continue                                   # not part of this split at all
        path = located.get(pdb)
        if path is None:
            missing_pdb_file.add(pdb)
            continue
        entry = dict(entry)
        entry['pdb_data_path'] = path                  # the stored path is from the old host
        buckets[split].append(entry)
    # MEAN reads this line by line (`for line in ...: json.loads(line)` in
    # MEAN/data/dataset.py), so it must be JSONL — a JSON array would fail to parse.
    for split in SPLITS:
        with open(os.path.join(out_dir, f'{split}.json'), 'w') as handle:
            for entry in buckets[split]:
                handle.write(json.dumps(entry) + '\n')
    return {s: len(buckets[s]) for s in SPLITS}, missing_pdb_file


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mean-summary', default='MEAN/summaries/sabdab_all.json')
    ap.add_argument('--pdb-dirs', nargs='*',
                    default=['MEAN/summaries/pdb', 'MEAN/summaries/all_nano_structures'])
    args = ap.parse_args()

    records, assignment = load_split(args.split_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    report = []

    ours = {s: {e['pdb'].lower() for e in records[s]} for s in SPLITS}
    report.append(f'split: {args.split_dir}')
    report.append('records  ' + '  '.join(f'{s}={len(records[s])}' for s in SPLITS))
    report.append('unique PDBs  ' + '  '.join(f'{s}={len(ours[s])}' for s in SPLITS))

    with open(os.path.join(args.output_dir, 'pdb_assignment.csv'), 'w') as handle:
        handle.write('pdb,split\n')
        for pdb in sorted(assignment):
            handle.write(f'{pdb},{assignment[pdb]}\n')
    report.append(f'wrote pdb_assignment.csv  ({len(assignment)} PDBs, none spanning splits)')

    counts = write_refinegnn(records, os.path.join(args.output_dir, 'refinegnn'))
    report.append('RefineGNN  ' + '  '.join(f'{s}={counts[s]}' for s in SPLITS) +
                  '   (full coverage: built from the same records)')

    if os.path.exists(args.mean_summary):
        entries = load_mean_summary(args.mean_summary)
        counts, missing = write_mean(entries, assignment, args.pdb_dirs,
                                     os.path.join(args.output_dir, 'mean'))
        report.append('MEAN       ' + '  '.join(f'{s}={counts[s]}' for s in SPLITS))
        covered = {s: len({p for p in ours[s] if p in assignment}) for s in SPLITS}
        for split in SPLITS:
            have = counts[split]
            want = len(ours[split])
            report.append(f'  {split:5s} PDB coverage {have}/{want} '
                          f'= {100.0 * have / want:.1f}%' if want else f'  {split}: empty')
        absent = sorted(p for p in assignment if p not in
                        {e['pdb'].lower() for e in entries})
        report.append(f'  PDBs absent from the MEAN summary : {len(absent)}')
        report.append(f'  PDBs with no local .pdb file      : {len(missing)}')
        with open(os.path.join(args.output_dir, 'missing_for_mean.txt'), 'w') as handle:
            handle.write('\n'.join(sorted(set(absent) | missing)) + '\n')
        report.append('  -> ids listed in missing_for_mean.txt (fetch from RCSB to close the gap)')

        # Fair-comparison test set: the complexes EVERY baseline can process. Table 1 must
        # be computed over exactly this subset for all methods, otherwise the methods are
        # scored on different data.
        with open(os.path.join(args.output_dir, 'mean', 'test.json')) as handle:
            mean_test = {json.loads(line)['pdb'].lower() for line in handle if line.strip()}
        evaluable = sorted(p for p in ours['test'] if p in mean_test)
        with open(os.path.join(args.output_dir, 'eval_subset_test_pdbs.txt'), 'w') as handle:
            handle.write('\n'.join(evaluable) + '\n')
        kept = sum(1 for e in records['test'] if e['pdb'].lower() in mean_test)
        report.append(f'  fair-comparison test subset: {len(evaluable)}/{len(ours["test"])} PDBs, '
                      f'{kept}/{len(records["test"])} records '
                      f'({100.0 * kept / len(records["test"]):.1f}%) '
                      f'-> eval_subset_test_pdbs.txt')
    else:
        report.append(f'MEAN summary not found at {args.mean_summary}; skipped')

    text = '\n'.join(report)
    with open(os.path.join(args.output_dir, 'coverage.txt'), 'w') as handle:
        handle.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
