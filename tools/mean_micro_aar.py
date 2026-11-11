#!/usr/bin/env python
"""Recompute MEAN's AAR as a MICRO-average, so it can be compared with the other methods.

MEAN reports a macro-average: generate.py:81 computes hit/len per complex and :185 (or
:140-147 for --rabd_test) averages those, weighting every complex equally. inference.py
:99-100 and RefineGNN/rabd_test.py:75-76 instead accumulate succ/tot over all positions --
a micro-average, weighted by CDR3 length. CDR3 lengths here span 5-26, so the two differ.
Table 1 must use one convention throughout; micro is the majority one.

    python tools/mean_micro_aar.py <results_dir> --test-json baselines/<split>/mean/test.json

<results_dir> is what generate.py wrote: either round dirs 0..N-1 (average_test) or top-k
dirs 0..k-1 (rabd_test). Each contains summary.json (JSONL, one record per complex).
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results_dir')
    ap.add_argument('--test-json', required=True,
                    help="MEAN test.json, to read each complex's native CDR3 length")
    args = ap.parse_args()

    lengths = {}
    with open(args.test_json) as fin:
        for line in fin:
            line = line.strip()
            if line:
                e = json.loads(line)
                lengths[e['pdb'].lower()] = len(e['cdrh3_seq'])

    summaries = sorted(glob.glob(os.path.join(args.results_dir, '*', 'summary.json')))
    if not summaries:
        raise SystemExit(f'no */summary.json under {args.results_dir}')

    grand_num = grand_den = 0.0
    for path in summaries:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        num = den = 0.0
        macro = 0.0
        for r in rows:
            L = lengths[r['pdb'].lower()]
            num += r['AAR'] * L
            den += L
            macro += r['AAR']
        grand_num += num
        grand_den += den
        tag = os.path.basename(os.path.dirname(path))
        print(f'  [{tag}] n={len(rows):4d}  macro={100*macro/len(rows):6.2f} %  '
              f'micro={100*num/den:6.2f} %')

    print(f'\npooled over {len(summaries)} group(s): '
          f'MICRO-AAR = {100 * grand_num / grand_den:.2f} %')


if __name__ == '__main__':
    main()
