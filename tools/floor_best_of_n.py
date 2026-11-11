#!/usr/bin/env python
"""The no-model floor UNDER THE SAME best-of-N protocol the methods are scored with.

Why this exists
---------------
Table 1 scores LmPro-Nano and RefineGNN on the best 100 of 10 000 samples ranked by the
model's own perplexity (`inference.py --num_decode 10000 --topk 100`;
`RefineGNN/rabd_test.py`). `tools/reference_baselines.py` reports single-shot floors. Those
two are not comparable: picking the most confident 1 % of a large sample is itself worth
some AAR, and until that is measured, "method X beats the 24 % floor" is uninterpretable.

This tool applies the identical protocol to a no-model baseline:

  1. fit a per-position residue frequency model on training CDR3s of the same length
     (the same model `reference_baselines.py` calls "positional-mode");
  2. draw --num-decode candidates i.i.d. per position from it;
  3. rank them by their log-likelihood under that same model -- the baseline's own
     "confidence", standing in for the network's perplexity;
  4. keep the top --topk and report micro-AAR over all kept positions.

Micro-averaging (sum of matches / sum of positions) matches inference.py:99-100 and
rabd_test.py:75-76. MEAN's generate.py macro-averages instead; see
docs/REPRODUCIBILITY.md, Baseline inputs.

    python tools/floor_best_of_n.py --dir abgnn/finetune/exp2-hern/nanobody_clustered \
        --restrict-pdbs baselines/nanobody_clustered/eval_subset_final.txt

Reading the result
------------------
An independent per-position model's most likely sequence IS the per-position mode, so if
the selected candidates collapse back onto the mode the best-of-N floor will sit close to
the single-shot floor -- which would be evidence that the protocol does NOT inflate a
trivial baseline, and that the methods' selection gains are real. If instead it climbs
well above the single-shot floor, then the single-shot floor is the wrong bar and every
AAR in Table 1 must be requoted against this one. Either outcome is a direct, data-driven
answer to reviewer R2.2.
"""
import argparse
import collections
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_baselines import positional_mode                          # noqa: E402


def load(directory, split, keep=None, records=None):
    path = os.path.join(directory, f'{split}_data.jsonl')
    out = []
    with open(path) as handle:
        for lineno, line in enumerate(handle):
            e = json.loads(line)
            if records is not None and lineno not in records:
                continue
            if keep is not None and e['pdb'].lower() not in keep:
                continue
            seq, tag = e['antibody_seq'], e['antibody_cdr']
            out.append(''.join(a for a, t in zip(seq, tag) if t == '3'))
    return out


def position_model(train, length, pseudo=1e-9):
    """Per-position residue distribution over training CDR3s of this length.

    Falls back to the whole training set for lengths never seen, exactly as
    reference_baselines.py's positional-mode baseline does.
    """
    pool = [c for c in train if len(c) == length] or train
    dists = []
    for i in range(length):
        counts = collections.Counter(c[i] for c in pool if i < len(c))
        total = sum(counts.values())
        if total == 0:
            counts, total = collections.Counter({'G': 1}), 1
        letters = sorted(counts)
        probs = [counts[a] / total for a in letters]
        dists.append((letters, probs, {a: math.log(counts[a] / total + pseudo) for a in letters}))
    return dists


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--restrict-pdbs', default=None,
                    help='PDB-level subset; pins PDBs but not records')
    ap.add_argument('--restrict-records', default=None,
                    help='record-level subset (eval_subset_records.txt) -- PREFERRED, pins '
                         'the identical records the methods are scored on')
    ap.add_argument('--num-decode', type=int, default=10000)
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--curve', action='store_true',
                    help='also report intermediate (N, k) points, to compare against the '
                         'methods measured at the same budgets')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.restrict_pdbs and args.restrict_records:
        raise SystemExit('pass only one of --restrict-pdbs / --restrict-records')
    keep = records = None
    if args.restrict_pdbs:
        with open(args.restrict_pdbs) as fin:
            keep = {l.strip().lower() for l in fin if l.strip()}
    if args.restrict_records:
        with open(args.restrict_records) as fin:
            records = {int(l.split('#', 1)[0].split()[0])
                       for l in fin if l.split('#', 1)[0].strip()}

    train = load(args.dir, 'train')
    test = load(args.dir, 'test', keep, records)
    print(f'{args.dir}   train={len(train)}  test={len(test)}'
          f'{"  (restricted)" if (keep or records) else ""}')

    budgets = [(args.num_decode, args.topk)]
    if args.curve:
        budgets = [(10, 2), (20, 2), (100, 10), (1000, 10), (args.num_decode, args.topk)]

    cache = {}
    for num_decode, topk in budgets:
        # Reseed per budget. Otherwise the earlier budgets in --curve consume the stream and
        # the same (N, k) reports a different value depending on whether --curve was passed
        # (observed: 22.44 vs 22.41 %). Each budget is now independently reproducible.
        rng = random.Random(args.seed)
        matched = total = 0
        single_matched = single_total = 0
        for native in test:
            L = len(native)
            if L == 0:
                continue
            if L not in cache:
                cache[L] = position_model(train, L)
            dists = cache[L]

            # single-shot reference. Uses reference_baselines.positional_mode() verbatim so
            # the two tools cannot disagree on the same published number.
            mode = positional_mode(train, L)
            single_matched += sum(a == b for a, b in zip(native, mode))
            single_total += L

            scored = []
            for _ in range(num_decode):
                chars, ll = [], 0.0
                for letters, probs, logp in dists:
                    c = rng.choices(letters, weights=probs, k=1)[0]
                    chars.append(c)
                    ll += logp[c]
                scored.append((ll, ''.join(chars)))
            scored.sort(key=lambda t: -t[0])          # most likely first
            for _ll, cand in scored[:topk]:
                matched += sum(a == b for a, b in zip(native, cand))
                total += L

        print(f'  best {topk:>4} of {num_decode:>6}  ->  micro-AAR = '
              f'{100.0 * matched / total:.2f} %')
    print(f'  single-shot positional mode (same records) = '
          f'{100.0 * single_matched / single_total:.2f} %')


if __name__ == '__main__':
    main()
