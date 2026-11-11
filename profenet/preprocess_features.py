# preprocess_features.py (Final Version - Multi-Stream .npy generation with all bug fixes)
import argparse
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import logging

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

# 假设您的项目结构允许这样导入
from rde.datasets.pdbredo_chain import PDBRedoChainDataset
from rde.geometry.curvature import compute_curvature_tensors, compute_shape_index
from rde.geometry.geodesic import construct_laplace_beltrami_operator, compute_heat_kernel_signature
from rde.utils.protein.constants import BBHeavyAtom, AA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocess_features')

def process_sample(sample, config):
    """
    为单个样本提取多模态特征，并分别保存为 .npy 文件。
    """
    try:
        if sample is None:
            return "Skipped: Sample is None."
            
        pdb_id_full = sample['id']
        processed_dir = config['data']['processed_dir']

        # 文件路径使用 .npy 后缀
        geo_path = os.path.join(processed_dir, f"{pdb_id_full}_geometry.npy")
        dihedral_path = os.path.join(processed_dir, f"{pdb_id_full}_dihedral.npy")
        local_env_path = os.path.join(processed_dir, f"{pdb_id_full}_local_env.npy")

        if os.path.exists(geo_path) and os.path.exists(dihedral_path) and os.path.exists(local_env_path):
            return f"Skipped: All feature files already exist for {pdb_id_full}."

        MAX_RESIDUES = 2000
        if sample['aa'].shape[0] > MAX_RESIDUES:
            num_res = sample['aa'].shape[0]
            return f"Skipped: Protein too large ({num_res} > {MAX_RESIDUES}) for {pdb_id_full}."

        # 从 sample 字典中提取所有需要的数据
        pos_heavyatom = sample['pos_heavyatom']
        mask_heavyatom = sample['mask_heavyatom']
        ca_coords = pos_heavyatom[:, BBHeavyAtom.CA, :].numpy()
        ca_mask = mask_heavyatom[:, BBHeavyAtom.CA].numpy().astype(bool)

        phi = sample['phi'].numpy()
        psi = sample['psi'].numpy()
        
        bfactor_ca = sample['bfactor_heavyatom'][:, BBHeavyAtom.CA].numpy()
        
        num_res = ca_coords.shape[0]

        # --- 1. 主链二面角特征 ---
        dihedral_features = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=1).astype(np.float32)
        np.save(dihedral_path, dihedral_features)

        # --- 2. 局部环境特征 (B-factor + 堆积密度) ---
        bfactor_ca_masked = bfactor_ca[ca_mask]
        if bfactor_ca_masked.size > 0:
            bfactor_norm = (bfactor_ca_masked - np.mean(bfactor_ca_masked)) / (np.std(bfactor_ca_masked) + 1e-6)
        else:
            bfactor_norm = np.array([])
        exposure_features = np.zeros((num_res, 1), dtype=np.float32)
        if bfactor_norm.size > 0:
            exposure_features[ca_mask] = bfactor_norm[:, np.newaxis]

        cb_coords = np.where(
            sample['aa'].numpy()[:, None] == AA.GLY,
            pos_heavyatom.numpy()[:, BBHeavyAtom.CA, :],
            pos_heavyatom.numpy()[:, BBHeavyAtom.CB, :]
        )
        cb_mask = np.where(
            sample['aa'].numpy() == AA.GLY,
            mask_heavyatom.numpy()[:, BBHeavyAtom.CA],
            mask_heavyatom.numpy()[:, BBHeavyAtom.CB]
        ).astype(bool)

        heavy_coords = pos_heavyatom[mask_heavyatom].numpy()
        if heavy_coords.shape[0] < 5:
             return f"Skipped: Not enough heavy atoms for {pdb_id_full}"

        tree = KDTree(heavy_coords)
        
        valid_cb_coords = cb_coords[cb_mask]
        radii = [4.0, 6.0, 8.0]
        
        if valid_cb_coords.shape[0] > 0:
            packing_counts = np.array([
                [len(indices) for indices in tree.query_ball_point(valid_cb_coords, r=r)] for r in radii
            ]).T.astype(np.float32)
            packing_counts = packing_counts - 1
            packing_norm = (packing_counts - np.mean(packing_counts, axis=0)) / (np.std(packing_counts, axis=0) + 1e-6)
        else:
            packing_norm = np.empty((0, len(radii)), dtype=np.float32)

        packing_features = np.zeros((num_res, len(radii)), dtype=np.float32)
        if packing_norm.shape[0] > 0:
            packing_features[cb_mask] = packing_norm

        local_env_features = np.concatenate([exposure_features, packing_features], axis=1).astype(np.float32)
        np.save(local_env_path, local_env_features)

        # --- 3. 几何特征 ---
        valid_ca_coords = ca_coords[ca_mask]
        if valid_ca_coords.shape[0] < 20: 
            return f"Skipped: Too few valid CA atoms for geometry in {pdb_id_full}."
            
        geo_cfg = config['features']['geometry']
        mean_curv, gauss_curv, princ_curvs = compute_curvature_tensors(valid_ca_coords, k=geo_cfg['curvature_neighbors'])
        shape_idx = compute_shape_index(princ_curvs)
        
        laplacian = construct_laplace_beltrami_operator(valid_ca_coords, k=geo_cfg['curvature_neighbors'])
        n_eigenvalues_req = geo_cfg['hks']['n_eigenvectors']
        n_eigenvalues_safe = min(n_eigenvalues_req, valid_ca_coords.shape[0] - 2)
        if n_eigenvalues_safe < 10: 
            return f"Skipped: Too small for HKS calculation in {pdb_id_full}."
        _, _, hks = compute_heat_kernel_signature(laplacian, n_eigenvalues=n_eigenvalues_safe, time_points=np.array(geo_cfg['hks']['timesteps']))
        
        geometry_features_valid = np.concatenate([mean_curv[:, None], gauss_curv[:, None], shape_idx[:, None], hks], axis=1)
        geometry_features = np.zeros((num_res, geometry_features_valid.shape[1]), dtype=np.float32)
        geometry_features[ca_mask] = geometry_features_valid
        np.save(geo_path, geometry_features)

        return f"Successfully processed {pdb_id_full}."

    except Exception as e:
        import traceback
        return f"Failed to process {sample.get('id', 'Unknown')}: {repr(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Pre-process features for the RDE model.")
    parser.add_argument('config', type=str, help="Path to the config YAML file.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers.")
    parser.add_argument('--test_subset', type=int, default=None, help="Run on a small subset for testing.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
    logger.info("Loading dataset for preprocessing...")
    dataset = PDBRedoChainDataset(
        split='all',
        pdbredo_dir=config['data']['pdbredo_dir'],
        clusters_path=config['data']['clusters_path'],
        splits_path=config['data']['splits_path'],
        processed_dir=config['data']['processed_dir'],
        reset=False
    )
    logger.info(f"Dataset loaded, containing {len(dataset)} chains to process.")
    
    samples_to_process = []
    processed_ids = set()
    for i in tqdm(range(len(dataset)), desc="Gathering unique samples"):
        try:
            sample = dataset[i]
            if sample is None: 
                continue
            
            pdb_id = sample['id']
            if pdb_id not in processed_ids:
                samples_to_process.append(sample)
                processed_ids.add(pdb_id)
        except Exception as e:
            logger.warning(f"Could not get sample for index {i}: {e}")
            continue
    
    if args.test_subset:
        logger.warning(f"--- RUNNING IN TEST MODE ON A SUBSET OF {args.test_subset} SAMPLES ---")
        samples_to_process = samples_to_process[:args.test_subset]

    logger.info(f"Total unique samples to process: {len(samples_to_process)}")

    results = Parallel(n_jobs=args.num_workers)(
        delayed(process_sample)(sample, config) for sample in tqdm(samples_to_process, desc="Processing features")
    )

    success_count = sum(1 for r in results if r and r.startswith('Successfully'))
    skipped_count = sum(1 for r in results if r and r.startswith('Skipped'))
    failed_count = sum(1 for r in results if r and r.startswith('Failed'))
    logger.info(f"Preprocessing finished. Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}.")
    
    if failed_count > 0:
        logger.warning("Listing some failed samples:")
        failed_samples = [r for r in results if r and r.startswith('Failed')]
        for i, r in enumerate(failed_samples):
            if i >= 10: break
            logger.warning(r)


if __name__ == '__main__':
    main()
