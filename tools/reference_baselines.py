#!/usr/bin/env python
"""No-model reference baselines for CDR3 design on a given split.

A design method is only interesting to the extent it beats these. They need no training
and no GPU, so they can be reported alongside Table 1 to tell the reader what the floor
is on whatever split is used -- which is exactly what makes a contaminated split visible
(on the original random split the nearest-framework copy baseline reaches 79.5% AAR).

    python tools/reference_baselines.py --dir abgnn/finetune/exp2-hern/nanobody_clustered

Baselines
  random-background   sample each position i.i.d. from the training CDR3 residue frequencies
  positional-mode     most frequent residue per position, over training CDR3s of the same length
  length-mode         most frequent training CDR3 of the same length (verbatim)
  nearest-framework   copy the CDR3 of the training record with the most similar framework
  nearest-antigen     copy the CDR3 of the training record with the most similar antigen

AAR is per-position identity against the native CDR3, averaged over all test positions
(the same convention inference.py uses).
"""
import argparse
import collections
import json
import os
import random
import sys


def load(directory, split, keep=None, records=None):
    """keep:    optional set of lowercase PDB ids to restrict to.
    records: optional set of 0-based source line indices (eval_subset_records.txt) --
             the record-level subset, preferred, see tools/record_ids.py.
    Both restrict the test set only; the baselines still draw on the whole training set,
    which is what a floor should do."""
    path = os.path.join(directory, f'{split}_data.jsonl')
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as handle:
        for lineno, line in enumerate(handle):
            e = json.loads(line)
            if records is not None and lineno not in records:
                continue
            if keep is not None and e['pdb'].lower() not in keep:
                continue
            seq, tag = e['antibody_seq'], e['antibody_cdr']
            out.append({
                'pdb': e['pdb'],
                'cdr3': ''.join(a for a, t in zip(seq, tag) if t == '3'),
                'framework': ''.join(a for a, t in zip(seq, tag) if t != '3'),
                'antigen': e.get('antigen_seq', ''),
            })
    return out


def positional_mode(train_cdr3s, length):
    """The per-position most common residue over training CDR3s of exactly this length.

    THE single definition of the positional-mode baseline -- tools/floor_best_of_n.py
    imports this rather than reimplementing it, because two implementations of the same
    published number that disagree by a tie-break (they did: 24.05 vs 24.15 %) is a defect.

    Falls back to the overall most common residue, repeated, when no training CDR3 has this
    length.
    """
    pool = [c for c in train_cdr3s if len(c) == length]
    if not pool:
        overall = collections.Counter(a for c in train_cdr3s for a in c).most_common(1)[0][0]
        return overall * length
    return ''.join(collections.Counter(c[i] for c in pool).most_common(1)[0][0]
                   for i in range(length))


def aar(pairs):
    """pairs: iterable of (native, predicted); predicted may be shorter/longer."""
    matched = total = 0
    for native, predicted in pairs:
        if predicted:
            matched += sum(a == b for a, b in zip(native, predicted))
        total += len(native)
    return 100.0 * matched / total if total else float('nan')


def most_similar(query, candidates, key):
    """Index of the candidate maximising normalised identity against query[key]."""
    best_score, best = -1.0, None
    q = query[key]
    for c in candidates:
        r = c[key]
        longer = max(len(q), len(r))
        if longer == 0:
            continue
        score = sum(a == b for a, b in zip(q, r)) / longer
        if score > best_score:
            best_score, best = score, c
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--restrict-records', '--restrict_records', dest='restrict_records',
                    default=None,
                    help='file of source-jsonl line indices (baselines/<split>/'
                         'eval_subset_records.txt). PREFERRED over --restrict-pdbs: pins '
                         'the identical records the methods are scored on (271, not 274).')
    ap.add_argument('--restrict-pdbs', '--restrict_pdbs', dest='restrict_pdbs', default=None,
                    help='file of PDB ids, one per line (baselines/<split>/'
                         'eval_subset_final.txt). Restricts the TEST set only. Pins the '
                         'PDBs but not the records -- see --restrict-records.')
    args = ap.parse_args()

    if args.restrict_pdbs and args.restrict_records:
        sys.exit('pass only one of --restrict-pdbs / --restrict-records')

    keep = records = None
    if args.restrict_pdbs:
        with open(args.restrict_pdbs) as fin:
            keep = {line.strip().lower() for line in fin if line.strip()}
    if args.restrict_records:
        with open(args.restrict_records) as fin:
            records = {int(l.split('#', 1)[0].split()[0])
                       for l in fin if l.split('#', 1)[0].strip()}

    train, test = load(args.dir, 'train'), load(args.dir, 'test', keep, records)
    if not train or not test:
        sys.exit(f'need train and test jsonl under {args.dir}')
    rng = random.Random(args.seed)

    if records:
        note = f'  (test restricted to {len(records)} records in {args.restrict_records})'
    elif keep:
        note = f'  (test restricted to {len(keep)} PDBs in {args.restrict_pdbs})'
    else:
        note = ''
    print(f'{args.dir}   train={len(train)} test={len(test)}{note}')
    print('  NOTE: these are SINGLE-SHOT floors. A method scored best-of-N must be compared '
          'against\n        tools/floor_best_of_n.py at the same (N, k) -- see '
          'docs/REPRODUCIBILITY.md, Baseline inputs.')

    # background residue frequencies over training CDR3s
    counts = collections.Counter(a for r in train for a in r['cdr3'])
    residues, weights = zip(*counts.items())
    print(f'  {"random-background":<20}'
          f'AAR = {aar((t["cdr3"], rng.choices(residues, weights, k=len(t["cdr3"]))) for t in test):5.2f}%')

    # per-length positional mode -- shared definition, see positional_mode()
    train_cdr3s = [r['cdr3'] for r in train]
    by_len = collections.defaultdict(list)
    for c in train_cdr3s:
        by_len[len(c)].append(c)
    print(f'  {"positional-mode":<20}'
          f'AAR = {aar((t["cdr3"], positional_mode(train_cdr3s, len(t["cdr3"]))) for t in test):5.2f}%')

    # most frequent whole CDR3 of the same length
    len_mode = {length: collections.Counter(seqs).most_common(1)[0][0]
                for length, seqs in by_len.items()}
    print(f'  {"length-mode":<20}'
          f'AAR = {aar((t["cdr3"], len_mode.get(len(t["cdr3"]), "")) for t in test):5.2f}%')

    # nearest neighbour copies
    for label, key in (('nearest-framework', 'framework'), ('nearest-antigen', 'antigen')):
        if key == 'antigen' and not any(r['antigen'] for r in train):
            continue
        pairs = []
        for t in test:
            hit = most_similar(t, train, key)
            pairs.append((t['cdr3'], hit['cdr3'] if hit else ''))
        print(f'  {label:<20}AAR = {aar(pairs):5.2f}%')


if __name__ == '__main__':
    main()
