#!/usr/bin/env python
"""Build namespaced ProFE-Net indices for a filtered PDB-REDO split.

The large LMDB and per-chain feature arrays are shared with the original
PDB-REDO cache. This script remaps the already validated original chains onto
the filtered cluster/split files without rescanning hundreds of thousands of
feature arrays.
"""

import argparse
import collections
import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path


def parse_sections(path):
    sections = collections.OrderedDict()
    current = None
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1]
                sections[current] = []
                continue
            if current is None:
                raise ValueError(f'Entry before section header in {path}: {line}')
            sections[current].extend(line.split())
    return sections


def parse_clusters(path):
    raw = parse_sections(path)
    clusters = collections.OrderedDict()
    for cluster, entries in raw.items():
        members = []
        for entry in entries:
            pdbcode, chain = entry.split(':', maxsplit=1)
            members.append((pdbcode, chain))
        clusters[cluster] = members
    return clusters


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_pickle(payload, path):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with tmp.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def flatten(split_clusters, sanitized_clusters):
    return [
        member
        for cluster in split_clusters
        for member in sanitized_clusters.get(cluster, [])
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='./data/pdbredo_processed_final')
    parser.add_argument('--source-splits', default='./data/pdbredo_splits.txt')
    parser.add_argument(
        '--filtered-clusters',
        default='./data/generated/pdbredo_skempi50_disjoint/pdbredo_clusters.txt',
    )
    parser.add_argument(
        '--filtered-splits',
        default='./data/generated/pdbredo_skempi50_disjoint/pdbredo_splits.txt',
    )
    parser.add_argument('--namespace', default='skempi50_disjoint_v1')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    source_sanitized_path = processed_dir / 'sanitized_clusters.pkl'
    with source_sanitized_path.open('rb') as handle:
        source_sanitized = pickle.load(handle)

    source_splits = parse_sections(args.source_splits)
    filtered_raw_clusters = parse_clusters(args.filtered_clusters)
    filtered_splits = parse_sections(args.filtered_splits)

    filtered_sanitized = collections.defaultdict(list)
    missing_cluster_names = []
    removed_unavailable_members = 0
    for cluster, members in filtered_raw_clusters.items():
        if cluster not in source_sanitized:
            missing_cluster_names.append(cluster)
            continue
        available = set(source_sanitized[cluster])
        retained = [member for member in members if member in available]
        removed_unavailable_members += len(members) - len(retained)
        if retained:
            filtered_sanitized[cluster] = retained

    sanitized_output = processed_dir / f'sanitized_clusters_{args.namespace}.pkl'
    output_paths = [sanitized_output]
    split_audit = {}
    valid_outputs = {}
    for split in ('train', 'val'):
        source_flat = flatten(source_splits[split], source_sanitized)
        source_indices_path = processed_dir / f'valid_indices_{split}.pkl'
        with source_indices_path.open('rb') as handle:
            source_valid_indices = pickle.load(handle)
        source_valid_members = {source_flat[index] for index in source_valid_indices}

        filtered_flat = flatten(filtered_splits[split], filtered_sanitized)
        filtered_valid_indices = [
            index
            for index, member in enumerate(filtered_flat)
            if member in source_valid_members
        ]
        output_path = processed_dir / f'valid_indices_{split}_{args.namespace}.pkl'
        output_paths.append(output_path)
        valid_outputs[split] = (output_path, filtered_valid_indices)
        split_audit[split] = {
            'source_flat_chains': len(source_flat),
            'source_valid_chains': len(source_valid_indices),
            'filtered_flat_chains': len(filtered_flat),
            'filtered_valid_chains': len(filtered_valid_indices),
        }

    audit_path = processed_dir / f'cache_audit_{args.namespace}.json'
    output_paths.append(audit_path)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            'Refusing to overwrite existing cache files without --force: '
            + ', '.join(existing)
        )

    atomic_pickle(filtered_sanitized, sanitized_output)
    for _, (output_path, indices) in valid_outputs.items():
        atomic_pickle(indices, output_path)

    audit = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'namespace': args.namespace,
        'source_sanitized_clusters': str(source_sanitized_path.resolve()),
        'source_splits': str(Path(args.source_splits).resolve()),
        'filtered_clusters': str(Path(args.filtered_clusters).resolve()),
        'filtered_splits': str(Path(args.filtered_splits).resolve()),
        'hashes': {
            'source_sanitized_clusters': sha256(source_sanitized_path),
            'source_splits': sha256(args.source_splits),
            'filtered_clusters': sha256(args.filtered_clusters),
            'filtered_splits': sha256(args.filtered_splits),
        },
        'filtered_raw_clusters': len(filtered_raw_clusters),
        'filtered_sanitized_clusters': len(filtered_sanitized),
        'missing_cluster_names': missing_cluster_names,
        'removed_unavailable_members': removed_unavailable_members,
        'splits': split_audit,
    }
    tmp_audit = audit_path.with_name(f'.{audit_path.name}.tmp-{os.getpid()}')
    with tmp_audit.open('w') as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    os.replace(tmp_audit, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
