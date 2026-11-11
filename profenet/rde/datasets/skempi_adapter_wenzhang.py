# rde/datasets/skempi_adapter.py (精简版: 只加载二面角特征)

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
from tqdm.auto import tqdm
import warnings

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# 假设这些自定义模块路径正确
from rde.utils.protein.constants import AA, restype_to_heavyatom_names, BBHeavyAtom

logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', PDBConstructionWarning)

class SkempiDataset(Dataset):
    # ... (此类保持不变, 和上一版完全相同) ...
    def __init__(self, csv_path, pdb_dir, cache_dir, num_cvfolds, cvfold_index, split='train'):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.cache_dir = os.path.join(cache_dir, f'fold{cvfold_index}_{split}')
        os.makedirs(self.cache_dir, exist_ok=True)
        df = pd.read_csv(csv_path, sep=';')
        df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed', 'Temperature'], inplace=True)
        df = df[df['Affinity_mut_parsed'] > 0]
        df = df[df['Affinity_wt_parsed'] > 0]
        df['Temperature'] = df['Temperature'].astype(str).str.extract(r'^(\d+\.?\d*)', expand=False)
        df.dropna(subset=['Temperature'], inplace=True)
        df['ddG'] = (8.314/4184) * (df['Temperature'].astype(float)) * np.log(df['Affinity_mut_parsed'] / df['Affinity_wt_parsed'])
        df['PDB_Code'] = df['#Pdb'].apply(lambda x: x.split('_')[0])
        unique_pdbs = sorted(list(df['PDB_Code'].unique()))
        fold_size = len(unique_pdbs) // num_cvfolds
        val_start = cvfold_index * fold_size
        val_end = (cvfold_index + 1) * fold_size if cvfold_index < num_cvfolds - 1 else len(unique_pdbs)
        val_pdbs = unique_pdbs[val_start:val_end]
        if split == 'train': self.df = df[~df['PDB_Code'].isin(val_pdbs)].reset_index(drop=True)
        elif split == 'val': self.df = df[df['PDB_Code'].isin(val_pdbs)].reset_index(drop=True)
        else: self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = os.path.join(self.cache_dir, f'data_{idx}.pt')
        if os.path.exists(cache_path):
            try: return torch.load(cache_path)
            except Exception: pass

        pdb_code, mutations_str, pdb_full_id = row['PDB_Code'], row['Mutation(s)_PDB'], row['#Pdb']
        protein_chains_str = pdb_full_id.split('_')[1:]
        pdb_path = os.path.join(self.pdb_dir, f'{pdb_code}.pdb')
        if not os.path.exists(pdb_path): return None

        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_code, pdb_path)
            model = structure[0]
            all_residues, chain_map = [], {}
            chain_group = "".join(protein_chains_str)
            for i, chain_id in enumerate(protein_chains_str):
                if chain_id in model and chain_id in chain_group:
                    chain_map[chain_id] = i
                    all_residues.extend([res for res in model[chain_id] if res.get_id()[0] == ' '])
            if not all_residues: return None
            
            n_residues = len(all_residues)
            pos_heavyatom = torch.zeros(n_residues, 15, 3, dtype=torch.float32)
            mask_heavyatom = torch.zeros(n_residues, 15, dtype=torch.bool)
            aa = torch.zeros(n_residues, dtype=torch.long)
            chain_nb = torch.zeros(n_residues, dtype=torch.long)
            mut_flag = torch.zeros(n_residues, dtype=torch.bool)
            res_nb = torch.zeros(n_residues, dtype=torch.long)
            res_idx_map = {res.get_full_id(): i for i, res in enumerate(all_residues)}

            for i, res in enumerate(all_residues):
                res_nb[i] = res.get_id()[1]
                resname = res.get_resname()
                try: aa_enum = AA(resname); aa[i] = aa_enum.value
                except ValueError: aa[i] = AA.UNK.value; continue
                chain_nb[i] = chain_map[res.get_parent().id]
                atom_names = restype_to_heavyatom_names.get(aa_enum, [])
                for atom_idx, atom_name in enumerate(atom_names):
                    if not atom_name or atom_idx >= 15: continue
                    if atom_name in res:
                        pos_heavyatom[i, atom_idx] = torch.from_numpy(res[atom_name].get_coord())
                        mask_heavyatom[i, atom_idx] = True
            
            aa_mut = aa.clone()
            for mut in mutations_str.split(','):
                wt_aa, chain_id, mut_aa_char = mut[0], mut[1], mut[-1]
                resseq_str = mut[2:-1]
                match = re.match(r'(-?\d+)([A-Za-z]?)', resseq_str)
                resseq, icode = int(match.group(1)), match.group(2).strip() or ' '
                found = False
                for res in all_residues:
                    if res.get_parent().id == chain_id and res.get_id() == (' ', resseq, icode):
                        res_idx = res_idx_map[res.get_full_id()]
                        mut_flag[res_idx] = True
                        aa_mut[res_idx] = AA(mut_aa_char).value
                        found = True
                        break
                if not found: logger.warning(f"Could not find residue for mutation {mut} in PDB {pdb_code}")

            data = {
                'id': pdb_code, 'chains': protein_chains_str, 'aa': aa, 'aa_mut': aa_mut, 'aa_wt': aa,
                'pos_atoms': pos_heavyatom, 'mask_atoms': mask_heavyatom, 'mask': mask_heavyatom[:, BBHeavyAtom.CA],
                'chain_nb': chain_nb, 'mut_flag': mut_flag, 'res_nb': res_nb,
                'ddG': torch.tensor(row['ddG'], dtype=torch.float32)
            }
            data['metadata'] = {'complex': row['#Pdb'], 'mutstr': row['Mutation(s)_PDB'], 'ddG': row['ddG']}
            torch.save(data, cache_path)
            return data
        except Exception as e:
            logger.error(f"Failed during BioPython processing of {pdb_code}: {e}", exc_info=False)
            return None

class SKEMPIDatasetAdapter(Dataset):
    def __init__(self, base_dataset, processed_dir):
        super().__init__()
        self.base_dataset = base_dataset
        self.processed_dir = processed_dir
        self.valid_indices = self._pre_filter()

    def _pre_filter(self):
        valid_indices = []
        for i in tqdm(range(len(self.base_dataset)), desc="Pre-filtering SKEMPI samples"):
            data = self.base_dataset[i]
            if data is None: continue
            
            chains_exist = True
            for chain_id in data['chains']:
                id_str = f"{data['id']}_{chain_id}"
                # ## MODIFICATION START: 只检查二面角特征文件 ##
                dihedral_path = os.path.join(self.processed_dir, f"{id_str}_dihedral.npz")
                if not os.path.exists(dihedral_path):
                    chains_exist = False
                    break
                # ## MODIFICATION END ##
            if chains_exist:
                valid_indices.append(i)
        
        logger.info(f"SKEMPI pre-filtering complete. Found {len(valid_indices)}/{len(self.base_dataset)} valid entries.")
        return valid_indices
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        data = self.base_dataset[original_idx]
        if data is None: return None
        
        try:
            all_dihedral_feats, all_phi, all_psi, all_phi_mask, all_psi_mask = [], [], [], [], []
            all_chi, all_chi_mask = [], []

            for chain_id in data['chains']:
                id_str = f"{data['id']}_{chain_id}"
                dihedral_path = os.path.join(self.processed_dir, f"{id_str}_dihedral.npz")

                # ## MODIFICATION START: 移除了对 geo_feat 及其 shape 的检查 ##
                # The check for geo_feat and its shape is no longer needed.
                # ## MODIFICATION END ##
                
                dihedral_data = np.load(dihedral_path)
                all_dihedral_feats.append(torch.from_numpy(dihedral_data['features']))
                all_phi.append(torch.from_numpy(dihedral_data['phi_raw']))
                all_psi.append(torch.from_numpy(dihedral_data['psi_raw']))
                all_phi_mask.append(torch.from_numpy(dihedral_data['phi_mask']))
                all_psi_mask.append(torch.from_numpy(dihedral_data['psi_mask']))
                all_chi.append(torch.from_numpy(dihedral_data['chi_raw']))
                all_chi_mask.append(torch.from_numpy(dihedral_data['chi_mask']))

            if not all_dihedral_feats: return None

            # ## MODIFICATION START: 移除对 geo 和 local_env 特征的赋值 ##
            # data['geometry_features'] = ...
            # data['local_env_features'] = ...
            # The lines above have been deleted.
            # ## MODIFICATION END ##

            data['dihedral_features'] = torch.cat(all_dihedral_feats, dim=0)
            data['phi'] = torch.cat(all_phi, dim=0)
            data['psi'] = torch.cat(all_psi, dim=0)
            data['phi_mask'] = torch.cat(all_phi_mask, dim=0)
            data['psi_mask'] = torch.cat(all_psi_mask, dim=0)
            data['chi'] = torch.cat(all_chi, dim=0)
            data['chi_mask'] = torch.cat(all_chi_mask, dim=0)
            
            return data
        except Exception as e:
            logger.error(f"Error loading features for index {idx} (original {original_idx}): {e}")
            return None