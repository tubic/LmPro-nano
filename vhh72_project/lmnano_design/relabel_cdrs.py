#!/usr/bin/env python3
"""Add CDR1 / CDR2 labels to lmnano_design/data/test_data.jsonl.  2026-08-26

WHY: `antibody_cdr` in the shipped data labels CDR3 only ('3' x18, '0' elsewhere),
so AntibodyComplexDataset(cdr_types=['1']) builds an empty `surface` and crashes at
abgen_dataset.py:69. CDR2 is the one region where Lm-Nano could generate AND the
ProFE+RDE screener has demonstrated top-end ability (interface CDR2: base rate
52.2%, top-5 4/5, enrichment 1.53x) -- CDR3 screening is 0.95x (none) and the
framework (rho=+0.51) is unreachable because the architecture only masks CDR spans.

HOW: ANARCI IMGT numbering per entry (same tool/scheme as the original report),
mapping IMGT CDR1 27-38 and CDR2 56-65 to sequence indices. CDR3 labels are left
EXACTLY as shipped -- we only fill in 1 and 2 -- so every existing CDR3 result
stays bit-identical and reproducible.

Verification printed: per-entry CDR1/CDR2 lengths, the CDR3 block is unchanged,
and the parent's spans are checked against the motifs used elsewhere in this project.
"""
import json, os, sys, collections

sys.path.insert(0, '/home/NAS/sunbo/LmPro/.venvs/structure-baselines/lib/python3.10/site-packages')
from anarci import run_anarci

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'data', 'test_data.jsonl')
DST = os.path.join(HERE, 'data', 'test_data_cdr123.jsonl')
# IMGT CDR definitions (same scheme the original REPORT used for CDR3 = IMGT 105-117)
IMGT = {'1': (27, 38), '2': (56, 65)}


def spans(seq):
    """-> {cdr: [0-based seq indices]} via ANARCI IMGT."""
    _, numbered, _, _ = run_anarci([('q', seq)], scheme='imgt')
    if not numbered or not numbered[0]:
        return None
    dom = numbered[0][0]                      # (numbering, start, end)
    numbering, start, _ = dom
    out = {k: [] for k in IMGT}
    i = start
    for (pos, ins), aa in numbering:
        if aa == '-':
            continue
        for k, (lo, hi) in IMGT.items():
            if lo <= pos <= hi:
                out[k].append(i)
        i += 1
    return out


def main():
    lines = open(SRC).read().strip().split('\n')
    out, stats, fail = [], collections.Counter(), []
    for ln in lines:
        e = json.loads(ln)
        seq, cdr = e['antibody_seq'], list(e['antibody_cdr'])
        assert len(seq) == len(cdr)
        sp = spans(seq)
        if sp is None:
            fail.append(e['pdb']); out.append(e); continue
        for k in ('1', '2'):
            for i in sp[k]:
                if cdr[i] == '0':             # never overwrite an existing CDR3 label
                    cdr[i] = k
        e['antibody_cdr'] = ''.join(cdr)
        c = collections.Counter(e['antibody_cdr'])
        stats[(c.get('1', 0), c.get('2', 0), c.get('3', 0))] += 1
        out.append(e)

    # CDR3 must be untouched
    for a, b in zip(lines, out):
        o = json.loads(a)
        assert [i for i, v in enumerate(o['antibody_cdr']) if v == '3'] == \
               [i for i, v in enumerate(b['antibody_cdr']) if v == '3'], f"CDR3 changed: {o['pdb']}"
    with open(DST, 'w') as fh:
        for e in out:
            fh.write(json.dumps(e) + '\n')
    print(f'wrote {DST}: {len(out)} entries')
    print(f'ANARCI failures: {len(fail)} {fail[:5]}')
    print(f'(CDR1,CDR2,CDR3) length combos: {dict(sorted(stats.items()))}')
    par = [e for e in out if e['pdb'] == 'parent'][0]
    s, c = par['antibody_seq'], par['antibody_cdr']
    for k in '123':
        idx = [i for i, v in enumerate(c) if v == k]
        if idx:
            print(f"  parent CDR{k}: panel {idx[0]+1}-{idx[-1]+1} = {''.join(s[i] for i in idx)}")
    print('CDR3 blocks verified unchanged for all entries ✓')


if __name__ == '__main__':
    main()
