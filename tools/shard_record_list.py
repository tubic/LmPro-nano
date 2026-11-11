#!/usr/bin/env python3
"""Split a commented record-index manifest into deterministic balanced shards."""

import argparse
from pathlib import Path


def load_records(path):
    records = []
    for raw in path.read_text().splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            records.append(int(value.split()[0]))
    if not records:
        raise ValueError(f"record list is empty: {path}")
    if len(records) != len(set(records)):
        raise ValueError(f"record list contains duplicates: {path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")

    records = load_records(args.input)
    if args.num_shards > len(records):
        parser.error("--num-shards cannot exceed the record count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = [records[index::args.num_shards] for index in range(args.num_shards)]
    for index, shard in enumerate(shards):
        output = args.output_dir / f"shard_{index:02d}.txt"
        output.write_text(
            f"# shard {index}/{args.num_shards}; source={args.input}\n"
            + "".join(f"{record}\n" for record in shard)
        )
        print(f"{output}: {len(shard)} records")


if __name__ == "__main__":
    main()
