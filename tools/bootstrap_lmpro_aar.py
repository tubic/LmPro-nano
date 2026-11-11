#!/usr/bin/env python3
"""Bootstrap LmPro micro-AAR with PDB clusters as the default sampling unit."""

import argparse
import json
from pathlib import Path

from paired_aar_bootstrap import bootstrap_aar, load_lmpro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmpro-tsv", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument(
        "--resampling-unit", choices=("pdb", "record"), default="pdb"
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.topk <= 0 or args.resamples <= 0:
        parser.error("--topk and --resamples must be positive")

    result = bootstrap_aar(
        load_lmpro(args.lmpro_tsv, args.topk),
        args.resamples,
        args.seed,
        args.resampling_unit,
    )
    result.update({"topk": args.topk, "lmpro_tsv": str(args.lmpro_tsv)})
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)


if __name__ == "__main__":
    main()
