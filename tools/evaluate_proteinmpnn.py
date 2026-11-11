#!/usr/bin/env python3
"""Evaluate ProteinMPNN CDR-H3 recovery on an exported record-level benchmark.

Candidates are ranked independently for each record by ProteinMPNN's own ``score``
(mean negative log probability over the designable positions).  AAR is a micro-average
over every retained candidate and native CDR-H3 position, matching the LmPro-Nano
top-k evaluation convention.
"""

import argparse
import csv
import json
import math
import os
import re


SAMPLE_RE = re.compile(r"(?:^|, )sample=(\d+)(?:,|$)")
SCORE_RE = re.compile(r"(?:^|, )score=([-+0-9.eE]+)(?:,|$)")


def read_fasta(path):
    records = []
    header = None
    sequence = []
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence)))
                header = line[1:]
                sequence = []
            elif header is None:
                raise ValueError(f"sequence before first FASTA header in {path}")
            else:
                sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def load_candidates(path):
    candidates = []
    for header, sequence in read_fasta(path):
        sample_match = SAMPLE_RE.search(header)
        if sample_match is None:
            continue  # first entry is the native sequence
        score_match = SCORE_RE.search(header)
        if score_match is None:
            raise ValueError(f"sample has no ProteinMPNN score in {path}: {header}")
        score = float(score_match.group(1))
        if not math.isfinite(score):
            raise ValueError(f"non-finite sample score in {path}: {header}")
        candidates.append((score, int(sample_match.group(1)), sequence.replace("/", "")))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seq-dir", required=True, help="ProteinMPNN output directory containing <name>.fa")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--json-out")
    parser.add_argument("--per-record-out")
    args = parser.parse_args()

    if any(k <= 0 for k in args.topk):
        parser.error("all --topk values must be positive")
    topks = sorted(set(args.topk))
    with open(args.manifest, newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    if not manifest:
        raise SystemExit("manifest is empty")
    backbone_conditions = {
        item.get("cdr_backbone_condition", "unspecified") for item in manifest
    }
    if len(backbone_conditions) != 1:
        raise SystemExit(
            f"manifest mixes CDR-H3 backbone conditions: {sorted(backbone_conditions)}"
        )
    backbone_condition = next(iter(backbone_conditions))

    rows = []
    minimum_candidates = None
    for item in manifest:
        path = os.path.join(args.seq_dir, item["name"] + ".fa")
        if not os.path.exists(path):
            raise SystemExit(f"missing ProteinMPNN output: {path}")
        candidates = load_candidates(path)
        if not candidates:
            raise SystemExit(f"no generated candidates in {path}")
        minimum_candidates = len(candidates) if minimum_candidates is None else min(
            minimum_candidates, len(candidates)
        )
        rows.append((item, candidates))

    unavailable = [k for k in topks if k > minimum_candidates]
    if unavailable:
        raise SystemExit(
            f"requested top-k {unavailable}, but at least one record has only "
            f"{minimum_candidates} candidates"
        )

    results = []
    per_record_results = []
    for topk in topks:
        matched = total = 0
        score_sum = 0.0
        for item, candidates in rows:
            start = int(item["cdr3_start"])
            end = int(item["cdr3_end"])
            native = item["cdr3_sequence"]
            native_full = item["antibody_sequence"]
            if len(native_full) != int(item["antibody_length"]):
                raise ValueError(f"invalid antibody length for {item['name']}")
            if len(native) != end - start or native_full[start:end] != native:
                raise ValueError(f"invalid CDR-H3 interval for {item['name']}")
            record_matched = record_total = 0
            record_score_sum = 0.0
            for score, _, sequence in candidates[:topk]:
                if len(sequence) != int(item["antibody_length"]):
                    raise ValueError(
                        f"generated length {len(sequence)} != antibody length "
                        f"{item['antibody_length']} for {item['name']}"
                    )
                if sequence[:start] != native_full[:start] or sequence[end:] != native_full[end:]:
                    raise ValueError(f"ProteinMPNN changed a fixed framework residue for {item['name']}")
                predicted = sequence[start:end]
                candidate_matches = sum(a == b for a, b in zip(native, predicted))
                matched += candidate_matches
                total += len(native)
                score_sum += score
                record_matched += candidate_matches
                record_total += len(native)
                record_score_sum += score
            per_record_results.append(
                {
                    "source_line": int(item["source_line"]),
                    "name": item["name"],
                    "pdb": item["pdb"],
                    "cdr_backbone_condition": backbone_condition,
                    "topk": topk,
                    "matched_positions": record_matched,
                    "total_positions": record_total,
                    "aar": record_matched / record_total,
                    "mean_proteinmpnn_score": record_score_sum / topk,
                }
            )
        result = {
            "topk": topk,
            "records": len(rows),
            "candidates_per_record": minimum_candidates,
            "selected_designs": len(rows) * topk,
            "matched_positions": matched,
            "total_positions": total,
            "aar": matched / total,
            "mean_proteinmpnn_score": score_sum / (len(rows) * topk),
        }
        results.append(result)
        print(
            f"top {topk:>4}: AAR = {100.0 * result['aar']:.2f}%  "
            f"({matched}/{total} positions; {len(rows)} records; "
            f"mean score={result['mean_proteinmpnn_score']:.4f})"
        )

    if args.per_record_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.per_record_out)), exist_ok=True)
        with open(args.per_record_out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_record_results[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(per_record_results)
        print(f"wrote {args.per_record_out}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(
                {
                    "manifest": args.manifest,
                    "seq_dir": args.seq_dir,
                    "cdr_backbone_condition": backbone_condition,
                    "results": results,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
