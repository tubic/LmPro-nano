# preprocess_features_skempi.py (简化版 - 移除冗余的geometry特征)

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
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from scipy.spatial import KDTree

# [简化] 移除了不再需要的geometry库导入
# from rde.geometry.curvature import compute_curvature_tensors, compute_shape_index
# from rde.geometry.geodesic import construct_laplace_beltrami_operator, compute_heat_kernel_signature
from rde.utils.protein.constants import BBHeavyAtom, AA, restype_to_heavyatom_names, chi_angles_atoms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocess_features_skempi_simplified')
warnings.simplefilter('ignore', PDBConstructionWarning)

MAX_RESIDUES = 2000

def process_chain(pdb_code, chain_id, pdb_dir, output_dir, config):
    pdb_path = os.path.join(pdb_dir, f'{pdb_code}.pdb')
    output_path = os.path.join(output_dir, f'{pdb_code}_{chain_id}.pt')

    if os.path.exists(output_path):
        return f"Skipped: {output_path} already exists."

    try:
        # 1. 加载结构
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_code, pdb_path)
        model = structure[0]
        if chain_id not in model: return f"Failed: Chain {chain_id} not in {pdb_code}."
        chain = model[chain_id]
        residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
        
        if len(residues) == 0: return f"Skipped: No standard residues in {pdb_code}_{chain_id}."
        if len(residues) > MAX_RESIDUES: return f"Skipped: Chain too large {pdb_code}_{chain_id}."

        n_residues = len(residues)
        
        # 2. 初始化张量
        #pos_heavyatom = torch.full([n_residues, 15, 3], torch.nan, dtype=torch.float32)
        pos_heavyatom = torch.zeros([n_residues, 15, 3], dtype=torch.float32)
        mask_heavyatom = torch.zeros([n_residues, 15], dtype=torch.bool)
        aa = torch.zeros(n_residues, dtype=torch.long)
        phi = torch.zeros(n_residues, dtype=torch.float32)
        psi = torch.zeros(n_residues, dtype=torch.float32)
        chi = torch.zeros(n_residues, 4, dtype=torch.float32)
        phi_mask = torch.zeros(n_residues, dtype=torch.bool)
        psi_mask = torch.zeros(n_residues, dtype=torch.bool)
        chi_mask = torch.zeros(n_residues, 4, dtype=torch.bool)
        res_map = {res.get_id(): i for i, res in enumerate(residues)}

        # 3. 提取原子坐标和氨基酸类型
        for i, res in enumerate(residues):
            resname = res.get_resname()
            try: aa_enum = AA(resname)
            except ValueError: aa[i] = AA.UNK.value; continue
            aa[i] = aa_enum.value
            atom_names = restype_to_heavyatom_names.get(aa_enum, [])
            for atom_idx, atom_name in enumerate(atom_names):
                if not atom_name: continue
                if atom_name in res:
                    pos_heavyatom[i, atom_idx] = torch.from_numpy(res[atom_name].get_coord())
                    mask_heavyatom[i, atom_idx] = True
        
        # 4. 提取Phi, Psi, Chi角和它们的掩码
        ppb = PPBuilder()
        polypeptides = ppb.build_peptides(chain)
        for poly in polypeptides:
            phi_psi_list = poly.get_phi_psi_list()
            for res_idx_in_poly, (res_phi, res_psi) in enumerate(phi_psi_list):
                res = poly[res_idx_in_poly]
                if res.get_id() in res_map:
                    i = res_map[res.get_id()]
                    if res_phi is not None: phi[i] = res_phi; phi_mask[i] = True
                    if res_psi is not None: psi[i] = res_psi; psi_mask[i] = True
            for res in poly:
                if res.get_id() not in res_map: continue
                i = res_map[res.get_id()]
                aa_enum = AA(res.get_resname())
                current_chi_defs = chi_angles_atoms.get(aa_enum, [])
                for chi_idx, atom_names in enumerate(current_chi_defs):
                    try:
                        v1, v2, v3, v4 = (res[name].get_vector() for name in atom_names)
                        chi[i, chi_idx] = calc_dihedral(v1, v2, v3, v4); chi_mask[i, chi_idx] = True
                    except KeyError: continue

        # 5. 组装数据字典
        sample = {
            'id': f'{pdb_code}_{chain_id}', 'aa': aa,
            'pos_heavyatom': pos_heavyatom, 'mask_heavyatom': mask_heavyatom,
            'phi': phi, 'psi': psi, 'chi': chi,
            'phi_mask': phi_mask, 'psi_mask': psi_mask, 'chi_mask': chi_mask
        }

        # 6. 计算并添加 Local Environment 特征
        ca_coords = pos_heavyatom[:, BBHeavyAtom.CA, :].numpy()
        ca_mask = mask_heavyatom[:, BBHeavyAtom.CA].numpy().astype(bool)
        heavy_coords = pos_heavyatom[mask_heavyatom].numpy()
        if heavy_coords.shape[0] < 5: return f"Skipped: Not enough heavy atoms for {pdb_code}_{chain_id}"
        tree = KDTree(heavy_coords)
        cb_coords = np.where(aa.numpy()[:, None] == AA.GLY, ca_coords, pos_heavyatom.numpy()[:, BBHeavyAtom.CB, :])
        cb_mask = np.where(aa.numpy() == AA.GLY, ca_mask, mask_heavyatom.numpy()[:, BBHeavyAtom.CB]).astype(bool)
        valid_cb_coords = cb_coords[cb_mask]
        radii = [4.0, 6.0, 8.0]
        if valid_cb_coords.shape[0] > 0:
            packing_counts = np.array([[len(indices) for indices in tree.query_ball_point(valid_cb_coords, r=r)] for r in radii]).T.astype(np.float32)
            packing_norm = (packing_counts - np.mean(packing_counts, axis=0)) / (np.std(packing_counts, axis=0) + 1e-6)
        else:
            packing_norm = np.zeros((0, len(radii)), dtype=np.float32)
        packing_features = np.zeros((n_residues, len(radii)), dtype=np.float32)
        if packing_norm.shape[0] > 0: packing_features[cb_mask] = packing_norm
        sample['local_env_features'] = torch.from_numpy(packing_features)

        # 7. [简化] 移除了完整的 geometry 特征计算模块
        
        # 8. 将所有特征统一保存到一个 .pt 文件
        torch.save(sample, output_path)
        return f"Successfully processed {pdb_code}_{chain_id}."

    except Exception as e:
        return f"Failed to process {pdb_code}_{chain_id}: {repr(e)}"

def main():
    parser = argparse.ArgumentParser(description="Pre-process features for SKEMPI and save to .pt files.")
    parser.add_argument('config', type=str, help="Path to the YAML config file.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    data_cfg = config['dataset']; pdb_dir = data_cfg['pdb_dir']; csv_path = data_cfg['csv_path']; output_dir = data_cfg['processed_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, sep=';')
    unique_structures = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing unique chains from CSV"):
        pdb_id_full = row.get('#Pdb', '');
        if not isinstance(pdb_id_full, str) or '_' not in pdb_id_full: continue
        pdb_code, *chain_groups = pdb_id_full.split('_')
        for group in chain_groups:
            for chain_id in group:
                unique_structures.add((pdb_code, chain_id))
    tasks = list(unique_structures)
    logger.info(f"Found {len(tasks)} unique protein chains to process.")
    results = Parallel(n_jobs=args.num_workers)(
        delayed(process_chain)(pdb_code, chain_id, pdb_dir, output_dir, config) 
        for pdb_code, chain_id in tqdm(tasks, desc="Processing chains")
    )
    success_count = sum(1 for r in results if r and r.startswith('Successfully'))
    skipped_count = sum(1 for r in results if r and r.startswith('Skipped'))
    failed_count = sum(1 for r in results if r and r.startswith('Failed'))
    logger.info(f"Preprocessing finished. Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}.")

if __name__ == '__main__':
    main()
