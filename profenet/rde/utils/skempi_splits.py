"""Deterministic SKEMPI split protocols and leakage audits."""

import math
import random
from collections import Counter


def _require_fields(entries, fields):
    for index, entry in enumerate(entries):
        missing = [field for field in fields if field not in entry]
        if missing:
            raise KeyError("Entry %d is missing fields: %s" % (index, missing))


def rde_complex_folds(entries, num_folds=3, seed=2022):
    """Reproduce the published RDE-PPI complex-shuffle protocol exactly."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    _require_fields(entries, ("complex",))
    complexes = sorted({str(entry["complex"]) for entry in entries})
    random.Random(seed).shuffle(complexes)
    fold_size = math.ceil(len(complexes) / num_folds)
    return [
        set(complexes[index * fold_size : (index + 1) * fold_size])
        for index in range(num_folds)
    ]


def balanced_pdb_folds(entries, num_folds=5, seed=2022):
    """Create PDB-disjoint folds balanced by row count without using labels."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    _require_fields(entries, ("pdbcode",))
    counts = Counter(str(entry["pdbcode"]) for entry in entries)
    rng = random.Random(seed)
    tie_breaker = {pdbcode: rng.random() for pdbcode in counts}
    pdbs = sorted(
        counts,
        key=lambda pdbcode: (-counts[pdbcode], tie_breaker[pdbcode], pdbcode),
    )

    folds = [set() for _ in range(num_folds)]
    row_counts = [0] * num_folds
    for pdbcode in pdbs:
        fold = min(
            range(num_folds),
            key=lambda index: (row_counts[index], len(folds[index]), index),
        )
        folds[fold].add(pdbcode)
        row_counts[fold] += counts[pdbcode]
    return folds


def audit_folds(entries, folds, group_field):
    """Validate one-time assignment and report complex/PDB overlap by fold."""
    if not folds:
        raise ValueError("At least one fold is required")
    _require_fields(entries, (group_field, "complex", "pdbcode"))

    owner = {}
    for fold_index, groups in enumerate(folds):
        for group in groups:
            group = str(group)
            if group in owner:
                raise ValueError(
                    "Group %s appears in folds %d and %d"
                    % (group, owner[group], fold_index)
                )
            owner[group] = fold_index

    fold_rows = [0] * len(folds)
    fold_complexes = [set() for _ in folds]
    fold_pdbs = [set() for _ in folds]
    unassigned = []
    for entry in entries:
        group = str(entry[group_field])
        if group not in owner:
            unassigned.append(group)
            continue
        fold = owner[group]
        fold_rows[fold] += 1
        fold_complexes[fold].add(str(entry["complex"]))
        fold_pdbs[fold].add(str(entry["pdbcode"]))
    if unassigned:
        raise ValueError("Unassigned groups: %s" % sorted(set(unassigned))[:10])

    complex_overlap = {}
    pdb_overlap = {}
    for left in range(len(folds)):
        for right in range(left + 1, len(folds)):
            pair = "%d-%d" % (left, right)
            shared_complexes = sorted(fold_complexes[left] & fold_complexes[right])
            shared_pdbs = sorted(fold_pdbs[left] & fold_pdbs[right])
            if shared_complexes:
                complex_overlap[pair] = shared_complexes
            if shared_pdbs:
                pdb_overlap[pair] = shared_pdbs

    return {
        "num_entries": len(entries),
        "num_folds": len(folds),
        "group_field": group_field,
        "fold_rows": fold_rows,
        "fold_complexes": [len(values) for values in fold_complexes],
        "fold_pdbs": [len(values) for values in fold_pdbs],
        "complex_overlap": complex_overlap,
        "pdb_overlap": pdb_overlap,
        "all_entries_assigned_once": True,
    }
