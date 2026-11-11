#!/usr/bin/env python3
"""Write source JSONL line numbers that survive LmPro-Nano's real data loader.

This is the authoritative way to align external baselines to LmPro-Nano's train or
validation records.  An optional input subset is intersected with the loader output;
this is used for the pretraining-exact-disjoint validation control.
"""

import argparse
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from record_ids import align_to_source, load_record_subset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-records", help="optional source-line subset to intersect")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--cdr-type", default="3")
    parser.add_argument("--epitope-size", type=int, default=20)
    args = parser.parse_args()

    from fairseq.data import Dictionary
    from fairseq_models.data.abgen_dataset import AntibodyComplexDataset

    seq_vocab = Dictionary.load(os.path.join(ROOT, "data", "seq_dict.txt"))
    tag_vocab = Dictionary.load(os.path.join(ROOT, "data", "tag_dict.txt"))
    mask_idx = seq_vocab.add_symbol("<mask>")
    dataset = AntibodyComplexDataset(
        data_path=args.split_dir,
        split=args.split,
        seq_vocab=seq_vocab,
        tag_vocab=tag_vocab,
        cdr_types=[args.cdr_type],
        L_target=args.epitope_size,
        pad_idx=seq_vocab.pad(),
        mask_idx=mask_idx,
        max_len=args.max_len,
    )
    source_jsonl = os.path.join(args.split_dir, f"{args.split}_data.jsonl")
    records = set(align_to_source(dataset.data, source_jsonl))
    before_intersection = len(records)
    if args.input_records:
        records &= load_record_subset(args.input_records)
    records = sorted(records)
    if not records:
        raise SystemExit("loader/input intersection is empty")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        handle.write(f"# LmPro-Nano-loadable {args.split} records; zero-based source JSONL lines\n")
        handle.write(f"# loader kept {before_intersection}; final intersection {len(records)}\n")
        for record in records:
            handle.write(f"{record}\n")
    print(f"loader kept {before_intersection} {args.split} records")
    if args.input_records:
        print(f"intersection with {args.input_records}: {len(records)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
