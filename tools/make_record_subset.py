#!/usr/bin/env python
"""Build the RECORD-level fair-comparison subset, so every method scores identical records.

`eval_subset_final.txt` pins the 184 PDBs every method can process, but on those PDBs
RefineGNN keeps 274 records and LmPro-Nano keeps 271 -- see tools/record_ids.py for why a
PDB list cannot express that difference and why the source line index is the identity.

This writes `eval_subset_records.txt`: the source-jsonl line indices that are BOTH
(a) inside the 184-PDB subset and (b) loadable by every method, i.e. the intersection of
RefineGNN's and LmPro-Nano's surviving records. Consume it with:

    python inference.py ... --restrict_records baselines/<split>/eval_subset_records.txt
    python third_party/RefineGNN/eval_subset.py ... --restrict_records <same file>

    python tools/make_record_subset.py --split-dir abgnn/finetune/exp2-hern/nanobody_clustered \
        --baseline-dir baselines/nanobody_clustered

Self-verifying: it re-checks the RefineGNN export alignment and exits non-zero if the
intersection is empty or the alignment cannot be trusted.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Running as `python tools/make_record_subset.py` puts tools/ on sys.path but NOT the repo
# root, so fairseq_models would not import. Add the root explicitly (it must also be the
# cwd -- abgen_task.py loads data/seq_dict.txt by relative path).
sys.path.insert(0, os.path.dirname(_HERE))
from record_ids import align_to_source, source_keys                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--baseline-dir', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--max-len', type=int, default=256, help="must match inference.py's")
    ap.add_argument('--cdr-type', default='3')
    args = ap.parse_args()

    src_jsonl = os.path.join(args.split_dir, f'{args.split}_data.jsonl')
    pdb_list = os.path.join(args.baseline_dir, 'eval_subset_final.txt')
    rg_jsonl = os.path.join(args.baseline_dir, 'refinegnn', f'{args.split}.jsonl')
    out_txt = os.path.join(args.baseline_dir, 'eval_subset_records.txt')
    out_tsv = os.path.join(args.baseline_dir, 'eval_subset_records.tsv')

    src = [json.loads(l) for l in open(src_jsonl) if l.strip()]
    with open(pdb_list) as fin:
        keep = {l.strip().lower() for l in fin if l.strip()}
    print(f'source records : {len(src)}')
    print(f'PDB subset     : {len(keep)} pdbs ({pdb_list})')

    # --- RefineGNN: verify the export really is 1:1 with the source, in order -----------
    rg = [json.loads(l) for l in open(rg_jsonl) if l.strip()]
    if len(rg) != len(src) or any(a['pdb'] != b['pdb'] or a['antibody_seq'] != b['seq']
                                  for a, b in zip(src, rg)):
        sys.exit(f'{rg_jsonl} is NOT line-for-line aligned with {src_jsonl}; the source '
                 f'line index is not a valid identity for RefineGNN and this subset would '
                 f'be wrong. Re-export with tools/export_baseline_splits.py.')
    rg_ok = set(range(len(src)))
    print(f'RefineGNN      : {len(rg_ok)} records (export verified 1:1 with source)')

    # --- LmPro-Nano: load through its real loader and align back to source lines --------
    from fairseq.data import Dictionary
    from fairseq_models.data.abgen_dataset import AntibodyComplexDataset
    seq_vocab = Dictionary.load(os.path.join('data', 'seq_dict.txt'))
    tag_vocab = Dictionary.load(os.path.join('data', 'tag_dict.txt'))
    mask_idx = seq_vocab.add_symbol('<mask>')
    ds = AntibodyComplexDataset(
        data_path=args.split_dir, split=args.split,
        seq_vocab=seq_vocab, tag_vocab=tag_vocab, cdr_types=[args.cdr_type],
        L_target=20, pad_idx=seq_vocab.pad(), mask_idx=mask_idx, max_len=args.max_len)
    lm_ok = set(align_to_source(ds.data, src_jsonl))
    print(f'LmPro-Nano     : {len(lm_ok)} records (aligned onto source line numbers)')

    # --- intersect ----------------------------------------------------------------------
    in_pdbs = {i for i, e in enumerate(src) if e['pdb'].lower() in keep}
    records = sorted(in_pdbs & rg_ok & lm_ok)
    dropped = sorted(in_pdbs - set(records))
    print(f'\nin the 184 PDBs           : {len(in_pdbs)}')
    print(f'  minus not-loadable      : {len(dropped)}  '
          f'{[(i, src[i]["pdb"]) for i in dropped]}')
    print(f'  -> COMMON RECORD SUBSET : {len(records)}')
    if not records:
        sys.exit('empty intersection -- refusing to write')

    pdbs = {src[i]['pdb'].lower() for i in records}
    with open(out_txt, 'w') as fout:
        fout.write(f'# record-level fair-comparison subset for {args.split_dir}\n')
        fout.write(f'# {len(records)} records over {len(pdbs)} PDBs; values are 0-based line\n')
        fout.write(f'# indices into {args.split}_data.jsonl. Built by tools/make_record_subset.py.\n')
        for i in records:
            fout.write(f'{i}\n')
    with open(out_tsv, 'w') as fout:
        fout.write('source_line\tpdb\tcdr3\n')
        for i in records:
            e = src[i]
            cdr3 = ''.join(a for a, t in zip(e['antibody_seq'], e['antibody_cdr']) if t == '3')
            fout.write(f'{i}\t{e["pdb"]}\t{cdr3}\n')
    print(f'\nwrote {out_txt}  ({len(records)} records, {len(pdbs)} PDBs)')
    print(f'wrote {out_tsv}')


if __name__ == '__main__':
    main()
