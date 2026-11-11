#!/usr/bin/env python3
"""Build a PDB-REDO split that is disjoint from usable SKEMPI structures.

The source cluster and split files are never modified. A complete sequence
cluster is removed when any of its member PDBs occurs in the usable SKEMPI
benchmark. Remaining validation and test members are then filtered with the
precedence train > val > test so that exact PDB identifiers do not cross the
generated splits.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_clusters(path):
    clusters = OrderedDict()
    current = None
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                if current in clusters:
                    raise ValueError("Duplicate cluster: %s" % current)
                clusters[current] = []
                continue
            if current is None:
                raise ValueError("Cluster member appears before a header")
            for token in line.split():
                pdbcode, chain = token.split(":", 1)
                clusters[current].append((pdbcode, chain))
    return clusters


def read_splits(path):
    splits = OrderedDict()
    current = None
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                if current in splits:
                    raise ValueError("Duplicate split: %s" % current)
                splits[current] = []
                continue
            if current is None:
                raise ValueError("Split member appears before a header")
            splits[current].extend(line.split())
    for required in ("train", "val", "test"):
        if required not in splits:
            raise ValueError("Missing split: %s" % required)
    return splits


def usable_skempi_pdbs(csv_path, pdb_dir, blocklist):
    pdbs = set()
    counts = {
        "raw_rows": 0,
        "blocked_rows": 0,
        "missing_structure_rows": 0,
        "nonfinite_affinity_rows": 0,
        "usable_rows": 0,
    }
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            counts["raw_rows"] += 1
            pdbcode = row["#Pdb"].split("_", 1)[0].upper()
            if pdbcode in blocklist:
                counts["blocked_rows"] += 1
                continue
            pdb_path = pdb_dir / (pdbcode + ".pdb")
            if not pdb_path.is_file():
                counts["missing_structure_rows"] += 1
                continue
            try:
                affinity_mut = float(row["Affinity_mut_parsed"])
                affinity_wt = float(row["Affinity_wt_parsed"])
            except (TypeError, ValueError):
                counts["nonfinite_affinity_rows"] += 1
                continue
            if not (
                math.isfinite(affinity_mut)
                and math.isfinite(affinity_wt)
                and affinity_mut > 0
                and affinity_wt > 0
            ):
                counts["nonfinite_affinity_rows"] += 1
                continue
            pdbs.add(pdbcode)
            counts["usable_rows"] += 1
    return pdbs, counts


def split_stats(clusters, names):
    names = [name for name in names if name in clusters]
    pairs = {
        (pdbcode.upper(), chain)
        for name in names
        for pdbcode, chain in clusters[name]
    }
    return {
        "clusters": len(names),
        "chain_members": len(pairs),
        "pdbs": len({pdbcode for pdbcode, _ in pairs}),
    }


def write_wrapped(handle, values, width=10):
    for start in range(0, len(values), width):
        handle.write(" ".join(values[start : start + width]) + "\n")


def write_clusters(path, clusters):
    with open(path, "w") as handle:
        for name, members in clusters.items():
            handle.write("[%s]\n" % name)
            write_wrapped(handle, ["%s:%s" % member for member in members])


def write_splits(path, splits):
    with open(path, "w") as handle:
        for name, cluster_names in splits.items():
            handle.write("[%s]\n" % name)
            write_wrapped(handle, cluster_names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="data/pdbredo_clusters.txt")
    parser.add_argument("--splits", default="data/pdbredo_splits.txt")
    parser.add_argument("--skempi-csv", default="data/SKEMPI_v2/skempi_v2.csv")
    parser.add_argument("--skempi-pdb-dir", default="data/SKEMPI_v2/PDBs")
    parser.add_argument(
        "--output-dir",
        default="data/generated/pdbredo_skempi50_disjoint",
    )
    parser.add_argument("--blocklist", nargs="*", default=["1KBH"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    clusters_path = Path(args.clusters).resolve()
    splits_path = Path(args.splits).resolve()
    skempi_csv = Path(args.skempi_csv).resolve()
    skempi_pdb_dir = Path(args.skempi_pdb_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "pdbredo_clusters.txt",
        output_dir / "pdbredo_splits.txt",
        output_dir / "excluded_skempi_pdbs.txt",
        output_dir / "excluded_clusters.txt",
        output_dir / "audit.json",
    ]
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError(
            "Output files already exist; pass --force to rebuild deterministically"
        )

    clusters = read_clusters(clusters_path)
    splits = read_splits(splits_path)
    skempi_pdbs, skempi_counts = usable_skempi_pdbs(
        skempi_csv, skempi_pdb_dir, {value.upper() for value in args.blocklist}
    )

    primary_splits = ("train", "val", "test")
    cluster_owners = {}
    for split in primary_splits:
        for cluster in splits[split]:
            previous = cluster_owners.setdefault(cluster, split)
            if previous != split:
                raise ValueError(
                    "Cluster %s occurs in both %s and %s" % (cluster, previous, split)
                )

    excluded_clusters = {
        name
        for name, members in clusters.items()
        if any(pdbcode.upper() in skempi_pdbs for pdbcode, _ in members)
    }
    filtered = OrderedDict(
        (name, list(members))
        for name, members in clusters.items()
        if name not in excluded_clusters
    )

    removed_for_pdb_disjoint = {"val": 0, "test": 0}
    claimed_pdbs = set()
    for split in primary_splits:
        split_names = set(splits[split])
        for name in list(filtered):
            if name not in split_names:
                continue
            members = filtered[name]
            if split != "train":
                kept = [
                    member
                    for member in members
                    if member[0].upper() not in claimed_pdbs
                ]
                removed_for_pdb_disjoint[split] += len(members) - len(kept)
                filtered[name] = kept
        claimed_pdbs.update(
            pdbcode.upper()
            for name in splits[split]
            if name in filtered
            for pdbcode, _ in filtered[name]
        )

    filtered = OrderedDict(
        (name, members) for name, members in filtered.items() if members
    )
    filtered_splits = OrderedDict(
        (
            split,
            [name for name in names if name in filtered],
        )
        for split, names in splits.items()
    )

    pdb_sets = {}
    chain_sets = {}
    for split in primary_splits:
        chain_sets[split] = {
            (pdbcode.upper(), chain)
            for name in filtered_splits[split]
            for pdbcode, chain in filtered[name]
        }
        pdb_sets[split] = {pdbcode for pdbcode, _ in chain_sets[split]}

    overlap_audit = {}
    for index, left in enumerate(primary_splits):
        for right in primary_splits[index + 1 :]:
            overlap_audit["%s-%s" % (left, right)] = {
                "pdb_overlap": len(pdb_sets[left] & pdb_sets[right]),
                "chain_overlap": len(chain_sets[left] & chain_sets[right]),
            }

    source_stats = {
        split: split_stats(clusters, splits[split]) for split in primary_splits
    }
    generated_stats = {
        split: split_stats(filtered, filtered_splits[split])
        for split in primary_splits
    }
    audit = {
        "schema_version": 1,
        "policy": {
            "skempi_filter": (
                "remove every existing sequence cluster containing a PDB from "
                "the usable SKEMPI benchmark"
            ),
            "split_pdb_filter": (
                "remove later-split members sharing an exact PDB with an earlier "
                "split, using train > val > test precedence"
            ),
            "homology_scope": (
                "existing cluster membership; the source cluster file is described "
                "as 50-percent identity, but its original MMseqs2 command is unavailable"
            ),
        },
        "source_hashes": {
            "clusters_sha256": sha256_file(clusters_path),
            "splits_sha256": sha256_file(splits_path),
            "skempi_csv_sha256": sha256_file(skempi_csv),
        },
        "skempi": {
            **skempi_counts,
            "usable_pdbs": len(skempi_pdbs),
        },
        "source_stats": source_stats,
        "generated_stats": generated_stats,
        "source_clusters": len(clusters),
        "generated_clusters": len(filtered),
        "excluded_clusters_for_skempi": len(excluded_clusters),
        "removed_members_for_split_pdb_disjoint": removed_for_pdb_disjoint,
        "split_overlap": overlap_audit,
    }

    write_clusters(output_dir / "pdbredo_clusters.txt", filtered)
    write_splits(output_dir / "pdbredo_splits.txt", filtered_splits)
    with open(output_dir / "excluded_skempi_pdbs.txt", "w") as handle:
        handle.write("\n".join(sorted(skempi_pdbs)) + "\n")
    with open(output_dir / "excluded_clusters.txt", "w") as handle:
        handle.write("\n".join(sorted(excluded_clusters)) + "\n")
    with open(output_dir / "audit.json", "w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
