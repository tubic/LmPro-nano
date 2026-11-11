#!/usr/bin/env python
"""Measure train/test information leakage in a nanobody-antigen split.

Reports, for every test record, whether the same CDR3 / PDB entry / full VHH sequence
also occurs in training, and runs a no-model "copy the CDR3 of the nearest-framework
training record" baseline. If that baseline scores well, the split is ranking
memorisation rather than design and cannot be used to compare methods.

    python tools/audit_leakage.py --dir abgnn/finetune/exp2-hern/nanobody
    python tools/audit_leakage.py --dir abgnn/finetune/exp2-hern/nanobody_clustered

Add --no-baseline to skip the (O(|test| x |train|)) copy baseline.
"""
import argparse
import json
import os
import sys


def load(directory, split):
    path = os.path.join(directory, f'{split}_data.jsonl')
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as handle:
        for line in handle:
            e = json.loads(line)
            seq, tag = e['antibody_seq'], e['antibody_cdr']
            out.append({
                'pdb': e['pdb'],
                'seq': seq,
                'cdr3': ''.join(a for a, t in zip(seq, tag) if t == '3'),
                'framework': ''.join(a for a, t in zip(seq, tag) if t != '3'),
                'antigen': e.get('antigen_seq', ''),
            })
    return out


def copy_baseline_aar(train, test):
    """AAR of copying the CDR3 of the training record with the most similar framework."""
    matched = total = 0
    for t in test:
        best_score, best_cdr3 = -1.0, None
        for r in train:
            longer = max(len(t['framework']), len(r['framework']))
            if longer == 0:
                continue
            score = sum(a == b for a, b in zip(t['framework'], r['framework'])) / longer
            if score > best_score:
                best_score, best_cdr3 = score, r['cdr3']
        if best_cdr3 is not None:
            matched += sum(a == b for a, b in zip(t['cdr3'], best_cdr3))
        total += len(t['cdr3'])
    return 100.0 * matched / total if total else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='directory holding {train,valid,test}_data.jsonl')
    ap.add_argument('--no-baseline', action='store_true')
    args = ap.parse_args()

    train, valid, test = (load(args.dir, s) for s in ('train', 'valid', 'test'))
    if not train or not test:
        sys.exit(f'need train and test jsonl under {args.dir}')

    print(f'{args.dir}')
    print(f'  sizes: train={len(train)} valid={len(valid)} test={len(test)}')

    tr_cdr3 = {r['cdr3'] for r in train}
    tr_pdb = {r['pdb'] for r in train}
    tr_seq = {r['seq'] for r in train}
    tr_ag = {r['antigen'] for r in train}
    n = len(test)

    def pct(hits):
        return f'{hits:5d}/{n}  = {100.0 * hits / n:5.1f}%'

    print('  test records whose ... also occurs in train:')
    print(f'    identical CDR3        {pct(sum(r["cdr3"] in tr_cdr3 for r in test))}')
    print(f'    same PDB entry        {pct(sum(r["pdb"] in tr_pdb for r in test))}')
    print(f'    identical VHH seq     {pct(sum(r["seq"] in tr_seq for r in test))}')
    print(f'    identical antigen     {pct(sum(r["antigen"] in tr_ag for r in test))}')

    everything = train + valid + test
    print('  redundancy over the whole dataset:')
    print(f'    records {len(everything)}, unique CDR3 {len({r["cdr3"] for r in everything})}, '
          f'unique VHH {len({r["seq"] for r in everything})}')

    if not args.no_baseline:
        aar = copy_baseline_aar(train, test)
        print(f'  no-model nearest-framework CDR3 copy baseline: AAR = {aar:.2f}%')
        if aar > 40:
            print('    ^ a trivial lookup scores this high: the split is contaminated and '
                  'cannot rank design methods.')


if __name__ == '__main__':
    main()
