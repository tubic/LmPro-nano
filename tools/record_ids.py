"""Stable record identity across the different loaders, for the fair-comparison subset.

The problem this solves
-----------------------
`eval_subset_final.txt` is a list of PDB ids. On the same 184 PDBs the loaders disagree on
how many *records* survive:

    RefineGNN  AntibodyDataset        -> 274   (1:1 with the source jsonl, drops nothing)
    LmPro-Nano AntibodyComplexDataset -> 271   (drops 1 of 4hjj and 2 of 6zqk:
                                                over-length / UNK_LIST / CDR-count filters)

So a PDB-level list cannot pin the two methods to the same records. A record-level identity
is needed, and the only stable one is the **0-based line index in the source
`<split>/test_data.jsonl`**, because:

  * RefineGNN's export (`baselines/*/refinegnn/test.jsonl`) is line-for-line aligned with
    that source file -- verified, 325/325, same pdb and same sequence at every position;
  * `AntibodyComplexDataset` reads the same file top to bottom and only ever *skips* lines
    (`abgen_dataset.py:53-60, :84`), never reorders them, so its surviving records can be
    aligned back onto source line numbers.

`(pdb, antibody_seq)` was considered and rejected as an identity: it is not unique
(280 distinct keys for 325 records; e.g. 7my2 and 1kxt each appear 3x with an identical
chain), so it cannot address an individual record.
"""
import json


def source_keys(jsonl_path):
    """(pdb, antibody_cdr) for every line of the source jsonl, in file order."""
    keys = []
    with open(jsonl_path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                e = json.loads(line)
                keys.append((e['pdb'], e['antibody_cdr']))
    return keys


def align_to_source(loaded, jsonl_path):
    """Map each surviving record of AntibodyComplexDataset back to its source line index.

    `loaded` is dataset.data. Each entry keeps `pdb` and `cdr_string` (the untouched
    `antibody_cdr` string, set at abgen_dataset.py:66) even though `antibody_seq` has been
    replaced by a tensor -- so those two fields are what the match uses.

    Both sequences are in source order and the loader only skips, so a single forward scan
    is exact even when several records share a (pdb, cdr) key.
    """
    keys = source_keys(jsonl_path)
    out, cursor = [], 0
    for rec in loaded:
        want = (rec['pdb'], rec['cdr_string'])
        while cursor < len(keys) and keys[cursor] != want:
            cursor += 1
        if cursor >= len(keys):
            raise ValueError(
                f'could not align loaded record {rec["pdb"]} onto {jsonl_path}; '
                f'the loader is no longer order-preserving and this mapping is invalid')
        out.append(cursor)
        cursor += 1
    if len(out) != len(loaded):
        raise ValueError('alignment lost records')
    if out != sorted(out) or len(set(out)) != len(out):
        raise ValueError('alignment is not strictly increasing -- refusing to use it')
    return out


def load_record_subset(path):
    """Read a record-subset file: one source line index per line, '#' comments allowed."""
    idx = set()
    with open(path) as fin:
        for line in fin:
            line = line.split('#', 1)[0].strip()
            if line:
                idx.add(int(line.split()[0]))
    return idx
