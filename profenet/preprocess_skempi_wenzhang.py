# preprocess_features_skempi.py (精简版：只处理二面角特征)

import argparse
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
import pandas as pd
import warnings

from Bio.PDB import PDBParser, calc_dihedral
from Bio.PDB.vectors import Vector
from Bio.PDB.Polypeptide import CaPPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# 假设这些自定义模块路径正确
from rde.utils.protein.constants import BBHeavyAtom, AA, restype_to_heavyatom_names, chi_angles_atoms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocess_features_skempi')
warnings.simplefilter('ignore', PDBConstructionWarning)

MAX_RESIDUES = 2000

def get_chi_angles(res, pos_heavyatom_map):
    chi_values = [0.0] * 4
    chi_mask = [False] * 4
    resname = res.get_resname()
    if resname not in chi_angles_atoms:
        return torch.tensor(chi_values), torch.tensor(chi_mask)
    for i, chi_atom_names in enumerate(chi_angles_atoms[resname]):
        atom_vectors = []
        all_atoms_present = True
        for atom_name in chi_atom_names:
            if atom_name in pos_heavyatom_map:
                atom_vectors.append(Vector(pos_heavyatom_map[atom_name]))
            else:
                all_atoms_present = False
                break
        if all_atoms_present:
            angle = calc_dihedral(atom_vectors[0], atom_vectors[1], atom_vectors[2], atom_vectors[3])
            chi_values[i] = angle
            chi_mask[i] = True
    return torch.tensor(chi_values), torch.tensor(chi_mask)

def load_structure_for_processing(pdb_path, chain_id):
    # ... (此函数保持不变, 和上一版完全相同) ...
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
        model = structure[0]
        if chain_id not in model: return None, f"Chain {chain_id} not found in {pdb_path}"
        chain = model[chain_id]
        residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
        if len(residues) == 0: return None, f"Chain {chain_id} in {pdb_path} contains no standard residues."
        if len(residues) > MAX_RESIDUES: return None, f"Chain {chain_id} in {pdb_path} too large."

        n_residues = len(residues)
        atom_storage_size = 15
        pos_heavyatom = torch.zeros(n_residues, atom_storage_size, 3, dtype=torch.float32)
        mask_heavyatom = torch.zeros(n_residues, atom_storage_size, dtype=torch.bool)
        bfactor_heavyatom = torch.zeros(n_residues, atom_storage_size, dtype=torch.float32)
        aa = torch.zeros(n_residues, dtype=torch.long)
        chi_tensor = torch.zeros(n_residues, 4, dtype=torch.float32)
        chi_mask_tensor = torch.zeros(n_residues, 4, dtype=torch.bool)

        for i, res in enumerate(residues):
            resname = res.get_resname()
            try: current_aa_enum = AA(resname)
            except ValueError:
                aa[i] = AA.UNK.value
                continue
            aa[i] = current_aa_enum.value
            
            atom_names = restype_to_heavyatom_names.get(current_aa_enum, [])
            pos_heavyatom_map_this_res = {}
            for atom_idx, atom_name in enumerate(atom_names):
                if not atom_name or atom_idx >= atom_storage_size: continue
                if atom_name in res:
                    atom = res[atom_name]
                    coord = torch.from_numpy(atom.get_coord())
                    pos_heavyatom[i, atom_idx] = coord
                    mask_heavyatom[i, atom_idx] = True
                    bfactor_heavyatom[i, atom_idx] = atom.get_bfactor()
                    pos_heavyatom_map_this_res[atom_name] = coord.numpy()

            chi_values, chi_mask = get_chi_angles(res, pos_heavyatom_map_this_res)
            chi_tensor[i] = chi_values
            chi_mask_tensor[i] = chi_mask

        peptides = CaPPBuilder().build_peptides(chain)
        phi_psi_list = [(res.get_id(), phi, psi) for poly in peptides for res, (phi, psi) in zip(poly, poly.get_phi_psi_list())]
        phi_psi_map = {res_id: (phi, psi) for res_id, phi, psi in phi_psi_list}
        phi_raw_tensor = torch.zeros(n_residues, dtype=torch.float32)
        psi_raw_tensor = torch.zeros(n_residues, dtype=torch.float32)
        phi_mask_tensor = torch.zeros(n_residues, dtype=torch.bool)
        psi_mask_tensor = torch.zeros(n_residues, dtype=torch.bool)

        for i, res in enumerate(residues):
            phi, psi = phi_psi_map.get(res.get_id(), (None, None))
            if phi is not None:
                phi_raw_tensor[i] = phi
                phi_mask_tensor[i] = True
            if psi is not None:
                psi_raw_tensor[i] = psi
                psi_mask_tensor[i] = True
        
        sample = {
            'id': f'{os.path.basename(pdb_path).replace(".pdb", "")}_{chain_id}',
            'aa': aa, 'pos_heavyatom': pos_heavyatom, 'mask_heavyatom': mask_heavyatom,
            'bfactor_heavyatom': bfactor_heavyatom,
            'phi_raw': phi_raw_tensor, 'psi_raw': psi_raw_tensor,
            'phi_mask': phi_mask_tensor, 'psi_mask': psi_mask_tensor,
            'chi_raw': chi_tensor, 'chi_mask': chi_mask_tensor,
        }
        return sample, "Success"
    except Exception as e:
        return None, f"Failed to parse {pdb_path} for chain {chain_id} with BioPython: {repr(e)}"


def process_sample(sample, config, processed_dir):
    try:
        if sample is None: return "Skipped: Sample is None."
        pdb_id_full = sample['id']
        dihedral_path = os.path.join(processed_dir, f"{pdb_id_full}_dihedral.npz")

        if os.path.exists(dihedral_path):
            return f"Skipped: Dihedral feature file already exists for {pdb_id_full}."

        phi_raw, psi_raw = sample['phi_raw'].numpy(), sample['psi_raw'].numpy()
        phi_mask, psi_mask = sample['phi_mask'].numpy(), sample['psi_mask'].numpy()
        chi_raw, chi_mask = sample['chi_raw'].numpy(), sample['chi_mask'].numpy()
        
        phi_proc, psi_proc = np.nan_to_num(phi_raw), np.nan_to_num(psi_raw)
        dihedral_features = np.stack([np.sin(phi_proc), np.cos(phi_proc), np.sin(psi_proc), np.cos(psi_proc)], axis=1).astype(np.float32)
        
        np.savez_compressed(
            dihedral_path,
            features=dihedral_features,
            phi_raw=phi_raw, psi_raw=psi_raw,
            phi_mask=phi_mask, psi_mask=psi_mask,
            chi_raw=chi_raw, chi_mask=chi_mask,
        )

        # ## MODIFICATION START: 移除了所有 local_env 和 geometry 特征的计算 ##
        # The entire block for calculating b-factor, packing density,
        # curvature, shape index, and HKS has been deleted.
        # ## MODIFICATION END ##
        
        return f"Successfully processed dihedral features for {pdb_id_full}."
    except Exception as e:
        return f"Failed to process {sample.get('id', 'Unknown')}: {repr(e)}"

# ... (main 函数保持不变) ...
def main():
    # ...
    # main function content is unchanged
    # ...
    print(">>> RUNNING SCRIPT VERSION: V7_LEAN_DIHEDRAL_ONLY <<<")
    parser = argparse.ArgumentParser(description="Pre-process features for the SKEMPI dataset using a config file.")
    parser.add_argument('config', type=str, help="Path to the YAML config file.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    data_cfg = config['data']
    pdb_dir, csv_path, output_dir = data_cfg['pdb_dir'], data_cfg['csv_path'], data_cfg['processed_dir']
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading SKEMPI CSV file from: {csv_path}")
    try: df = pd.read_csv(csv_path, sep=';')
    except FileNotFoundError: logger.error(f"Error: SKEMPI CSV file not found at {csv_path}."); return
    pdb_column_name = '#Pdb'
    if pdb_column_name not in df.columns: logger.error(f"错误: 在CSV文件中找不到列 '{pdb_column_name}'。可用列为: {df.columns.tolist()}"); return
    df['PDB_Code'] = df[pdb_column_name].apply(lambda x: x.split('_')[0])
    unique_structures = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing mutations from CSV"):
        pdb_code = row['PDB_Code']
        partners_str = row.get(pdb_column_name)
        if not isinstance(partners_str, str): continue
        chain_groups = partners_str.split('_')[1:]
        for group in chain_groups:
            for chain_id in group:
                if chain_id: unique_structures.add((pdb_code, chain_id))
    tasks = list(unique_structures)
    logger.info(f"Found {len(tasks)} unique protein chains to process from SKEMPI dataset.")
    results = Parallel(n_jobs=args.num_workers)(delayed(load_structure_for_processing)(os.path.join(pdb_dir, f'{task[0]}.pdb'), task[1]) for task in tqdm(tasks, desc="Loading PDB structures"))
    samples_to_process = [sample for sample, msg in results if sample is not None]
    logger.info(f"Successfully loaded {len(samples_to_process)} structures. Starting feature extraction...")
    final_results = Parallel(n_jobs=args.num_workers)(delayed(process_sample)(sample, config, output_dir) for sample in tqdm(samples_to_process, desc="Processing features"))
    success_count = sum(1 for r in final_results if r and r.startswith('Successfully'))
    skipped_count = sum(1 for r in final_results if r and r.startswith('Skipped'))
    failed_count = sum(1 for r in final_results if r and r.startswith('Failed'))
    logger.info(f"Preprocessing finished. Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}.")

if __name__ == '__main__':
    main()