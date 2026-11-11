#!/usr/bin/env python3
"""Build Lm-Nano dataset entries for the VHH72 generation-layer test.

Inputs (read-only):
  - vhh72_project/panel50.csv                      (parent / B_bli / A_alphaseq sequences)
  - vhh72_project/parental_complex_final.pdb       (full-length parent chain A + CoV-2 RBD chain E)
Outputs:
  - vhh72_project/lmnano_design/data/test_data.jsonl   (AntibodyComplexDataset format)
  - vhh72_project/lmnano_design/entries_meta.csv

CDR3 definition: ANARCI IMGT 105-117 (exclusive of anchors C104 / W118), the same
convention as the LmPro-Nano finetuning data (verified on 4hjj test record).

Coordinates: framework residues are copied from the parent chain A by IMGT-number
alignment; CDR3 residues get the parent's CDR3 coordinates when the variant CDR3 has
the parental length (16), otherwise a linear interpolation between the IMGT-104/118
anchor residues (the same interpolation the dataset itself applies for init coords).
The antigen (chain E, 196 aa, 6LZG/7x2l CoV-2 RBD) is shared by all entries.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)                      # vhh72_project/
ROOT = os.path.dirname(PROJ)                      # repo root
STRUCT_VENV_PY = os.path.join(ROOT, '.venvs', 'structure-baselines', 'bin', 'python')
HMMER_BIN = os.path.join(ROOT, '.venvs', 'hmmer', 'bin')

PANEL_CSV = os.path.join(PROJ, 'panel50.csv')
COMPLEX_PDB = os.path.join(PROJ, 'parental_complex_final.pdb')
OUT_JSONL = os.path.join(HERE, 'data', 'test_data.jsonl')
OUT_META = os.path.join(HERE, 'entries_meta.csv')

N_RANDOM = 100
RANDOM_SEED = 20260825
AA20 = 'ACDEFGHIKLMNPQRSTVWY'

RES3TO1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
           'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
           'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
           'TYR': 'Y', 'VAL': 'V'}
# atom14 slots per residue identity (1-letter), mirrors ab_dictionary._RES_ATOM14_BY_RES
RES_ATOM14 = {
    'G': ['N', 'CA', 'C', 'O'],
    'A': ['N', 'CA', 'C', 'O', 'CB'],
    'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
}


def parse_pdb_chain(path, chain_id):
    """Return (seq, resnums, coords) with coords a list per residue of {atom_name: xyz}."""
    res = {}
    order = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            if line[21] != chain_id:
                continue
            rs = int(line[22:26])
            rn = RES3TO1[line[17:20].strip()]
            an = line[12:16].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if rs not in res:
                res[rs] = {'resname': rn, 'atoms': {}}
                order.append(rs)
            res[rs]['atoms'][an] = xyz
    order.sort()
    seq = ''.join(res[r]['resname'] for r in order)
    return seq, order, [res[r]['atoms'] for r in order]


def atom14_from_atoms(res_id, atoms):
    """[14,3] array for a residue of identity res_id given {atom_name: xyz}."""
    out = np.zeros((14, 3))
    names = RES_ATOM14[res_id]
    for slot, an in enumerate(names):
        if an in atoms:
            out[slot] = atoms[an]
    return out


def run_anarci_imgt(seqs):
    """ANARCI IMGT numbering for a list of (name, seq); returns {name: [((n,ins),aa), ...]}."""
    driver = (
        "import json,sys\n"
        "from anarci import anarci\n"
        "seqs=json.load(open(sys.argv[1]))\n"
        "res=anarci(seqs, scheme='imgt', output=False, allowed_species=['human','mouse','alpaca'])\n"
        "out={}\n"
        "for (name,_),domains in zip(seqs,res[0]):\n"
        "    if not domains: out[name]=None; continue\n"
        "    numbered,start,end=domains[0]\n"
        "    out[name]={'start':start,'end':end,'numbered':[[n,ins,aa] for (n,ins),aa in numbered]}\n"
        "json.dump(out,open(sys.argv[2],'w'))\n"
    )
    tmp_in = os.path.join(HERE, 'data', '_anarci_in.json')
    tmp_out = os.path.join(HERE, 'data', '_anarci_out.json')
    with open(tmp_in, 'w') as fh:
        json.dump(seqs, fh)
    env = dict(os.environ, PATH=HMMER_BIN + ':' + os.environ['PATH'])
    subprocess.run([STRUCT_VENV_PY, '-c', driver, tmp_in, tmp_out], check=True, env=env)
    with open(tmp_out) as fh:
        out = json.load(fh)
    os.remove(tmp_in)
    os.remove(tmp_out)
    return {
        k: (None if v is None else {
            'start': v['start'], 'end': v['end'],
            'numbered': [((n, ins), aa) for n, ins, aa in v['numbered']],
        })
        for k, v in out.items()
    }


def main():
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)

    rows = list(csv.DictReader(open(PANEL_CSV)))
    parent_seq = [r for r in rows if r['track'] == 'anchor_parent'][0]['sequence']
    bli = [(r['source_id'], r['sequence'], r['kd_nM']) for r in rows if r['track'] == 'B_bli']
    panel_a = [(r['clone_id'], r['sequence']) for r in rows if r['track'] == 'A_alphaseq']

    pdb_ab_seq, _, pdb_ab_atoms = parse_pdb_chain(COMPLEX_PDB, 'A')
    ag_seq, _, ag_atoms = parse_pdb_chain(COMPLEX_PDB, 'E')
    assert len(pdb_ab_seq) == 125 and len(ag_seq) == 196
    # NOTE: PDB chain A carries the N84/S85-swapped variant of the panel parent.
    # We keep the panel parent as the canonical sequence (AlphaBind mutation
    # numbering is panel-consistent) and the PDB coordinates as context.

    # random controls: 2-3 uniform point mutations inside the parental CDR3
    rng = np.random.default_rng(RANDOM_SEED)

    # --- ANARCI IMGT on all sequences (parent first) ---
    named = [('parent', parent_seq)] + [(n, s) for n, s, _ in bli] + panel_a
    numbering = run_anarci_imgt(named)

    def imgt_maps(name, seq):
        dom = numbering[name]
        assert dom is not None, f'ANARCI failed on {name}'
        numbered, start, end = dom['numbered'], dom['start'], dom['end']
        # `numbered` lists every IMGT reference position of the aligned domain,
        # including gap ('-') entries where the query has no residue. Non-gap
        # entries map in order onto query residues start..end.
        res = [None] * len(seq)
        i = start
        for (n, ins), aa in numbered:
            if aa == '-':
                continue
            assert i <= end and seq[i] == aa, f'ANARCI/query mismatch at offset {i} for {name}'
            res[i] = (n, ins)
            i += 1
        assert i == end + 1, f'numbered domain does not cover query region for {name}'
        cdr3_idx = [k for k, m in enumerate(res) if m is not None and 105 <= m[0] <= 117]
        assert cdr3_idx, f'no IMGT CDR3 for {name}'
        return res, cdr3_idx[0], cdr3_idx[-1] + 1

    parent_map, p_l, p_r = imgt_maps('parent', parent_seq)
    parent_cdr3 = parent_seq[p_l:p_r]
    print(f'parent CDR3 span 0-based [{p_l},{p_r}) = 1-based {p_l+1}-{p_r}: {parent_cdr3} (len {p_r-p_l})')
    parent_by_imgt = {m: i for i, m in enumerate(parent_map) if m is not None}
    a104 = [i for i, m in enumerate(parent_map) if m and m[0] == 104][0]
    a118 = [i for i, m in enumerate(parent_map) if m and m[0] == 118][0]
    anchor_l = atom14_from_atoms(parent_seq[a104], pdb_ab_atoms[a104])
    anchor_r = atom14_from_atoms(parent_seq[a118], pdb_ab_atoms[a118])
    parent_cdr3_coords = [atom14_from_atoms(parent_seq[i], pdb_ab_atoms[i]) for i in range(p_l, p_r)]

    def interp_coords(n_res):
        """linear interpolation between CDR3 anchors, n_res residues (dataset-style)."""
        out = []
        for k in range(1, n_res + 1):
            out.append((anchor_l + (anchor_r - anchor_l) * (k / (n_res + 1))))
        return out

    def build_variant_coords(name, seq):
        vmap, l, r = imgt_maps(name, seq)
        cdr3_len = r - l
        coords = [None] * len(seq)
        if cdr3_len == len(parent_cdr3):
            cdr3_coords = parent_cdr3_coords
            cdr3_src = 'parent_cdr3_coords'
        else:
            cdr3_coords = interp_coords(cdr3_len)
            cdr3_src = f'interpolated_cdr3_len{cdr3_len}'
        for k, i in enumerate(range(l, r)):
            coords[i] = cdr3_coords[k]
        # framework: map by IMGT number to the parent residue
        n_interp = 0
        for i, m in enumerate(vmap):
            if coords[i] is not None:
                continue
            if m is not None:
                hit = [j for j, pm in enumerate(parent_map) if pm == m]
                if hit:
                    j = hit[0]
                    coords[i] = atom14_from_atoms(parent_seq[j], pdb_ab_atoms[j])
                    continue
            # insertion absent from parent: fill later by neighbour interpolation
            n_interp += 1
        # unnumbered heads/tails and unmatched insertions: nearest mapped neighbour
        mapped = [i for i, c in enumerate(coords) if c is not None]
        for i, c in enumerate(coords):
            if c is None:
                left = max([j for j in mapped if j < i], default=None)
                right = min([j for j in mapped if j > i], default=None)
                if left is not None and right is not None:
                    w = (i - left) / (right - left)
                    coords[i] = coords[left] + (coords[right] - coords[left]) * w
                elif left is not None:
                    coords[i] = coords[left]
                else:
                    coords[i] = coords[right]
        assert all(c is not None for c in coords)
        return np.stack(coords), l, r, cdr3_src, n_interp

    entries = []
    meta = []

    def add_entry(name, group, seq, kd=''):
        coords, l, r, cdr3_src, n_interp = build_variant_coords(name, seq)
        cdr_str = '0' * l + '3' * (r - l) + '0' * (len(seq) - r)
        entries.append({
            'pdb': name,
            'antibody_seq': seq,
            'antibody_coords': coords.tolist(),
            'antibody_cdr': cdr_str,
            'antigen_seq': ag_seq,
            'antigen_coords': [atom14_from_atoms(ag_seq[i], ag_atoms[i]).tolist() for i in range(len(ag_seq))],
        })
        meta.append({
            'name': name, 'group': group, 'kd_nM': kd,
            'cdr3_start_1based': l + 1, 'cdr3_end_1based': r, 'cdr3': seq[l:r],
            'len': len(seq), 'coord_source': cdr3_src, 'n_fw_interp': n_interp,
        })

    add_entry('parent', 'parent', parent_seq)
    for n, s, kd in bli:
        add_entry(n, 'bli_validated', s, kd)
    for n, s in panel_a:
        add_entry(n, 'panelA', s)

    # random controls share the parental span/coords (same length)
    l, r = p_l, p_r
    cdr3 = parent_cdr3
    rand_muts = []
    for k in range(N_RANDOM):
        n_mut = int(rng.integers(2, 4))
        pos = sorted(rng.choice(r - l, size=n_mut, replace=False).tolist())
        new = list(cdr3)
        for p in pos:
            choices = [a for a in AA20 if a != cdr3[p]]
            new[p] = choices[int(rng.integers(0, len(choices)))]
        new_cdr3 = ''.join(new)
        seq = parent_seq[:l] + new_cdr3 + parent_seq[r:]
        name = f'random_{k:03d}'
        mut_desc = ';'.join(f'{cdr3[p]}{p_l + p + 1}{new_cdr3[p]}' for p in pos)
        cdr_str = '0' * l + '3' * (r - l) + '0' * (len(seq) - r)
        entries.append({
            'pdb': name,
            'antibody_seq': seq,
            'antibody_coords': [atom14_from_atoms(parent_seq[i], pdb_ab_atoms[i]).tolist()
                                for i in range(len(parent_seq))],
            'antibody_cdr': cdr_str,
            'antigen_seq': entries[0]['antigen_seq'],
            'antigen_coords': entries[0]['antigen_coords'],
        })
        meta.append({'name': name, 'group': 'random_control', 'kd_nM': '',
                     'cdr3_start_1based': l + 1, 'cdr3_end_1based': r, 'cdr3': new_cdr3,
                     'len': len(seq), 'coord_source': mut_desc, 'n_fw_interp': 0})
        rand_muts.append((name, mut_desc))

    with open(OUT_JSONL, 'w') as fh:
        for e in entries:
            fh.write(json.dumps(e) + '\n')
    with open(OUT_META, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)

    from collections import Counter
    print('entries:', len(entries), Counter(m['group'] for m in meta))
    print('coord sources:', Counter(m['coord_source'] for m in meta if m['group'] != 'random_control'))
    print('wrote', OUT_JSONL)
    print('wrote', OUT_META)


if __name__ == '__main__':
    main()
