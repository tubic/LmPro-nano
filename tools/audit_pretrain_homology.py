#!/usr/bin/env python
"""Audit downstream CDR3 homology against the Lm-Nano pretraining corpus.

The existing exact-overlap audit is intentionally cheap and exact. This tool
adds the sequence-search control requested by the reviewers: it extracts CDR3
spans from the fairseq mmap corpus, deduplicates them, and searches downstream
test and validation CDR3s with the same identity/coverage settings used to make
the downstream clustered split.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from fairseq.data import Dictionary, data_utils

from audit_pretrain_overlap import load_record_subset, locate_cdr3_spans, native_cdr3


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_pretrain_corpus(pretrain_dir):
    pretrain_dir = Path(pretrain_dir)
    sequence_dictionary = Dictionary.load(str(pretrain_dir / "seq" / "dict.txt"))
    tag_dictionary = Dictionary.load(str(pretrain_dir / "tag" / "dict.txt"))
    sequence_dataset = data_utils.load_indexed_dataset(
        str(pretrain_dir / "seq" / "train"), sequence_dictionary, None
    )
    tag_dataset = data_utils.load_indexed_dataset(
        str(pretrain_dir / "tag" / "train"), tag_dictionary, None
    )
    if len(sequence_dataset) != len(tag_dataset) or not np.array_equal(
        sequence_dataset.sizes, tag_dataset.sizes
    ):
        raise SystemExit("pretraining sequence and tag datasets are not aligned")

    sequence_data = np.memmap(
        pretrain_dir / "seq" / "train.bin",
        mode="r",
        dtype=sequence_dataset._index._dtype,
    )
    tag_data = np.memmap(
        pretrain_dir / "tag" / "train.bin",
        mode="r",
        dtype=tag_dataset._index._dtype,
    )
    return (
        sequence_dictionary,
        tag_dictionary,
        sequence_dataset,
        tag_dataset,
        sequence_data,
        tag_data,
    )


def target_cache_matches(metadata_path, expected):
    if not metadata_path.exists():
        return False
    try:
        with metadata_path.open() as handle:
            observed = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    keys = ("complete", "source", "max_pretrain_spans")
    return all(observed.get(key) == expected.get(key) for key in keys)


def build_target_fasta(args):
    target_fasta = Path(args.target_fasta)
    metadata_path = Path(args.target_metadata)
    source = {
        key: file_fingerprint(Path(args.pretrain_dir) / relative)
        for key, relative in {
            "seq_bin": "seq/train.bin",
            "seq_idx": "seq/train.idx",
            "seq_dict": "seq/dict.txt",
            "tag_bin": "tag/train.bin",
            "tag_idx": "tag/train.idx",
            "tag_dict": "tag/dict.txt",
        }.items()
    }
    expected = {
        "complete": True,
        "source": source,
        "max_pretrain_spans": args.max_pretrain_spans,
    }
    if (
        not args.force_rebuild_target
        and target_fasta.exists()
        and target_cache_matches(metadata_path, expected)
    ):
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        print(
            f"reusing target FASTA with {metadata['unique_cdr3s']} unique CDR3s: "
            f"{target_fasta}",
            flush=True,
        )
        return metadata

    (
        sequence_dictionary,
        tag_dictionary,
        sequence_dataset,
        _tag_dataset,
        sequence_data,
        tag_data,
    ) = load_pretrain_corpus(args.pretrain_dir)
    starts, ends = locate_cdr3_spans(
        tag_data, tag_dictionary.index("3"), args.chunk_tokens
    )
    if args.max_pretrain_spans:
        starts = starts[: args.max_pretrain_spans]
        ends = ends[: args.max_pretrain_spans]
    lengths = ends - starts

    ascii_lookup = np.zeros(len(sequence_dictionary), dtype=np.uint8)
    for index in range(len(sequence_dictionary)):
        symbol = sequence_dictionary[index]
        if len(symbol) == 1 and symbol in STANDARD_AA:
            ascii_lookup[index] = ord(symbol)

    target_fasta.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_fasta.with_suffix(target_fasta.suffix + ".tmp")
    seen = set()
    unique_count = 0
    processed = 0
    started = time.time()
    sha256 = hashlib.sha256()
    with temporary.open("wb", buffering=8 * 1024 * 1024) as handle:
        for length in np.unique(lengths):
            matching_starts = starts[lengths == length]
            for begin in range(0, len(matching_starts), args.extract_batch_spans):
                offsets = matching_starts[begin : begin + args.extract_batch_spans]
                indices = offsets[:, None] + np.arange(length, dtype=np.int64)
                tokens = np.ascontiguousarray(sequence_data[indices], dtype=np.uint8)
                for row in tokens:
                    key = row.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    encoded = ascii_lookup[row]
                    if not encoded.all():
                        bad = sorted(
                            {
                                sequence_dictionary[int(token)]
                                for token in row[encoded == 0]
                            }
                        )
                        raise RuntimeError(
                            f"non-amino-acid token(s) inside a pretraining CDR3: {bad}"
                        )
                    entry = (
                        f">p{unique_count:09d}\n".encode("ascii")
                        + encoded.tobytes()
                        + b"\n"
                    )
                    handle.write(entry)
                    sha256.update(entry)
                    unique_count += 1
                processed += len(offsets)
                if processed % 500_000 < len(offsets):
                    print(
                        f"extracted {processed}/{len(starts)} spans, "
                        f"unique={unique_count}, elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
    os.replace(temporary, target_fasta)
    metadata = {
        **expected,
        "pretrain_records": len(sequence_dataset),
        "extracted_spans": len(starts),
        "unique_cdr3s": unique_count,
        "target_fasta": str(target_fasta.resolve()),
        "target_fasta_sha256": sha256.hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(metadata_path, metadata)
    print(
        f"wrote {unique_count} unique pretraining CDR3s to {target_fasta}", flush=True
    )
    return metadata


def load_group(split_dir, split, restrict_records):
    source_path = Path(split_dir) / f"{split}_data.jsonl"
    with source_path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    restricted = load_record_subset(restrict_records)
    selected = restricted if restricted is not None else set(range(len(records)))
    if not selected <= set(range(len(records))):
        raise SystemExit(f"record subset for {split} contains an out-of-range index")
    return source_path, records, selected


def write_queries(args):
    groups = {}
    sequence_to_query = {}
    query_to_sequence = {}
    for name, split, restrict_records in (
        ("test", "test", args.test_restrict_records),
        ("valid", "valid", args.valid_restrict_records),
    ):
        source_path, records, selected = load_group(
            args.split_dir, split, restrict_records
        )
        groups[name] = {
            "split": split,
            "source_path": str(source_path.resolve()),
            "restrict_records": (
                str(Path(restrict_records).resolve()) if restrict_records else None
            ),
            "records": records,
            "selected": selected,
        }
        for source_line in sorted(selected):
            sequence = native_cdr3(records[source_line])
            if not sequence or not set(sequence) <= STANDARD_AA:
                raise RuntimeError(
                    f"invalid {split} CDR3 at source line {source_line}: {sequence!r}"
                )
            if sequence not in sequence_to_query:
                query_id = f"q{len(sequence_to_query):06d}"
                sequence_to_query[sequence] = query_id
                query_to_sequence[query_id] = sequence

    query_fasta = Path(args.query_fasta)
    query_fasta.parent.mkdir(parents=True, exist_ok=True)
    with query_fasta.open("w") as handle:
        for query_id, sequence in query_to_sequence.items():
            handle.write(f">{query_id}\n{sequence}\n")
    with Path(args.query_manifest).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("query_id", "cdr3"))
        writer.writerows(query_to_sequence.items())
    return groups, sequence_to_query, query_to_sequence


def run_mmseqs(args):
    mmseqs = shutil.which(args.mmseqs) or args.mmseqs
    if not Path(mmseqs).is_file():
        raise SystemExit(f"MMseqs2 executable not found: {args.mmseqs}")
    result_path = Path(args.mmseqs_results)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.mmseqs_tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="audit_", dir=tmp_root))
    query_db = str(work_dir / "query")
    target_db = str(work_dir / "target")
    prefilter_db = str(work_dir / "prefilter")
    alignment_db = str(work_dir / "alignment")
    fields = (
        "query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,"
        "qcov,tcov,evalue,bits,tseq"
    )
    commands = [
        [
            mmseqs,
            "createdb",
            args.query_fasta,
            query_db,
            "--threads",
            str(args.threads),
        ],
        [
            mmseqs,
            "createdb",
            args.target_fasta,
            target_db,
            "--threads",
            str(args.threads),
        ],
        [
            mmseqs,
            "ungappedprefilter",
            query_db,
            target_db,
            prefilter_db,
            "--min-ungapped-score",
            "0",
            "--prefilter-mode",
            "1",
            "--max-seqs",
            str(args.max_seqs),
            "--comp-bias-corr",
            "0",
            "--threads",
            str(args.threads),
        ],
        [
            mmseqs,
            "align",
            query_db,
            target_db,
            prefilter_db,
            alignment_db,
            "--alignment-mode",
            "3",
            "--min-seq-id",
            str(args.min_seq_id),
            "-c",
            str(args.coverage),
            "--cov-mode",
            str(args.cov_mode),
            "-e",
            str(args.evalue),
            "--comp-bias-corr",
            "0",
            "--threads",
            str(args.threads),
        ],
        [
            mmseqs,
            "convertalis",
            query_db,
            target_db,
            alignment_db,
            args.mmseqs_results,
            "--format-output",
            fields,
            "--threads",
            str(args.threads),
        ],
    ]
    try:
        for command in commands:
            print("running: " + " ".join(command), flush=True)
            subprocess.run(command, check=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return commands


def normalized_fraction(value):
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def parse_hits(args):
    fields = (
        "query",
        "target",
        "pident",
        "alnlen",
        "qstart",
        "qend",
        "qlen",
        "tstart",
        "tend",
        "tlen",
        "qcov",
        "tcov",
        "evalue",
        "bits",
        "tseq",
    )
    best = {}
    hit_counts = collections.Counter()
    with Path(args.mmseqs_results).open() as handle:
        reader = csv.DictReader(handle, fieldnames=fields, delimiter="\t")
        for row in reader:
            query = row["query"]
            row["pident"] = normalized_fraction(row["pident"])
            row["qcov"] = normalized_fraction(row["qcov"])
            row["tcov"] = normalized_fraction(row["tcov"])
            row["bits"] = float(row["bits"])
            row["evalue"] = float(row["evalue"])
            hit_counts[query] += 1
            rank = (row["pident"], min(row["qcov"], row["tcov"]), row["bits"])
            current = best.get(query)
            if current is None or rank > current[0]:
                best[query] = (rank, row)
    return {query: row for query, (_rank, row) in best.items()}, hit_counts


def write_group_outputs(
    args, name, group, sequence_to_query, query_to_sequence, best_hits, hit_counts
):
    rows = []
    hit_records = []
    for source_line in sorted(group["selected"]):
        record = group["records"][source_line]
        cdr3 = native_cdr3(record)
        query_id = sequence_to_query[cdr3]
        hit = best_hits.get(query_id)
        if hit is not None:
            hit_records.append(source_line)
        rows.append(
            {
                "source_line": source_line,
                "pdb": record["pdb"],
                "query_id": query_id,
                "cdr3": cdr3,
                "cdr3_length": len(cdr3),
                "has_homology_hit": int(hit is not None),
                "reported_hits": hit_counts[query_id],
                "best_target": hit["target"] if hit else "",
                "best_target_cdr3": hit["tseq"] if hit else "",
                "best_identity": hit["pident"] if hit else "",
                "best_query_coverage": hit["qcov"] if hit else "",
                "best_target_coverage": hit["tcov"] if hit else "",
                "best_evalue": hit["evalue"] if hit else "",
                "best_bits": hit["bits"] if hit else "",
            }
        )

    per_record_path = Path(args.output_dir) / f"{name}_per_record.tsv"
    with per_record_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    output_subset = Path(
        args.test_disjoint_records if name == "test" else args.valid_disjoint_records
    )
    output_subset.parent.mkdir(parents=True, exist_ok=True)
    kept = [row["source_line"] for row in rows if not row["has_homology_hit"]]
    with output_subset.open("w") as handle:
        handle.write("# Pretraining-CDR3-homology-disjoint subset\n")
        handle.write(
            f"# Derived from {group['restrict_records'] or group['source_path']}\n"
        )
        handle.write(
            f"# MMseqs2 min_seq_id={args.min_seq_id}, coverage={args.coverage}, "
            f"cov_mode={args.cov_mode}; kept {len(kept)} of {len(rows)} records.\n"
        )
        for source_line in kept:
            handle.write(f"{source_line}\n")

    unique_queries = {row["query_id"] for row in rows}
    hit_queries = unique_queries & set(best_hits)
    return {
        "split": group["split"],
        "source_path": group["source_path"],
        "restrict_records": group["restrict_records"],
        "records": len(rows),
        "records_with_hit": len(hit_records),
        "records_disjoint": len(kept),
        "unique_cdr3s": len(unique_queries),
        "unique_cdr3s_with_hit": len(hit_queries),
        "unique_cdr3s_disjoint": len(unique_queries - hit_queries),
        "per_record_tsv": str(per_record_path.resolve()),
        "disjoint_records": str(output_subset.resolve()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", default="abgnn/final_3_data")
    parser.add_argument(
        "--split-dir", default="abgnn/finetune/exp2-hern/nanobody_clustered"
    )
    parser.add_argument(
        "--test-restrict-records",
        default="baselines/nanobody_clustered/eval_subset_records.txt",
    )
    parser.add_argument("--valid-restrict-records", default=None)
    parser.add_argument(
        "--output-dir",
        default="baselines/nanobody_clustered/pretrain_homology_audit",
    )
    parser.add_argument(
        "--test-disjoint-records",
        default="baselines/nanobody_clustered/eval_subset_records_pretrain_homology_disjoint.txt",
    )
    parser.add_argument(
        "--valid-disjoint-records",
        default="baselines/nanobody_clustered/valid_records_pretrain_homology_disjoint.txt",
    )
    parser.add_argument(
        "--mmseqs",
        default="/home/af3_bosun/miniconda3/envs/flowpacker/bin/mmseqs",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--min-seq-id", type=float, default=0.6)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=1)
    parser.add_argument("--evalue", type=float, default=1_000_000.0)
    parser.add_argument("--max-seqs", type=int, default=10_000)
    parser.add_argument("--minimum-test-hit-records", type=int, default=0)
    parser.add_argument("--minimum-valid-hit-records", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=50_000_000)
    parser.add_argument("--extract-batch-spans", type=int, default=100_000)
    parser.add_argument(
        "--max-pretrain-spans",
        type=int,
        default=0,
        help="truncate extraction for smoke tests; zero means the full corpus",
    )
    parser.add_argument("--force-rebuild-target", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.target_fasta = str(output_dir / "pretrain_cdr3_unique.fasta")
    args.target_metadata = str(output_dir / "pretrain_cdr3_unique.metadata.json")
    args.query_fasta = str(output_dir / "downstream_queries.fasta")
    args.query_manifest = str(output_dir / "downstream_queries.tsv")
    args.mmseqs_results = str(output_dir / "mmseqs_hits.tsv")
    args.mmseqs_tmp = str(
        Path(os.environ.get("TMPDIR", "/tmp")) / "lmpro_pretrain_homology_mmseqs"
    )
    return args


def main():
    args = parse_args()
    started = time.time()
    target_metadata = build_target_fasta(args)
    groups, sequence_to_query, query_to_sequence = write_queries(args)
    commands = run_mmseqs(args)
    best_hits, hit_counts = parse_hits(args)

    group_summaries = {}
    for name, group in groups.items():
        group_summaries[name] = write_group_outputs(
            args,
            name,
            group,
            sequence_to_query,
            query_to_sequence,
            best_hits,
            hit_counts,
        )
    required_hits = {
        "test": args.minimum_test_hit_records,
        "valid": args.minimum_valid_hit_records,
    }
    for name, minimum in required_hits.items():
        observed = group_summaries[name]["records_with_hit"]
        if observed < minimum:
            raise RuntimeError(
                f"{name} homology search found {observed} hit records, below the "
                f"required known-positive minimum of {minimum}"
            )
    summary = {
        "protocol": {
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "evalue": args.evalue,
            "max_seqs": args.max_seqs,
            "prefilter": "exhaustive_ungapped_no_composition_bias",
            "mmseqs_version": subprocess.check_output(
                [commands[0][0], "version"], text=True
            ).strip(),
            "commands": commands,
        },
        "target": target_metadata,
        "queries": len(query_to_sequence),
        "queries_with_hit": len(best_hits),
        "required_known_positive_hits": required_hits,
        "groups": group_summaries,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = Path(args.output_dir) / "summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
