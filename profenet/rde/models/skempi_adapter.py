# rde/datasets/skempi_adapter.py

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import prody
prody.confProDy(verbosity='none')
import re # 导入正则表达式库

from rde.utils.protein.constants import AA, restype_to_heavyatom_names, BBHeavyAtom

logger = logging.getLogger(__name__)

class SkempiDataset(Dataset):
    """
    负责从 SKEMPI v2.0 CSV 文件中解析数据，并根据PDB文件构建数据集。
    同时处理交叉验证的划分。
    """
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
        
        if split == 'train':
            self.df = df[~df['PDB_Code'].isin(val_pdbs)].reset_index(drop=True)
        elif split == 'val':
            self.df = df[df['PDB_Code'].isin(val_pdbs)].reset_index(drop=True)
        else: # 'all'
            self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = os.path.join(self.cache_dir, f'data_{idx}.pt')
        
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path)
            except (IOError, EOFError):
                pass

        pdb_code = row['PDB_Code']
        mutations_str = row['Mutation(s)_PDB']
        
        first_mut = mutations_str.split(',')[0]
        chain_id = first_mut[1]

        # =================================================================================
        # === 关键修正：使用正则表达式解析残基号和插入码(icode) ===
        # =================================================================================
        res_id_str = first_mut[2:-1]
        match = re.match(r'(-?\d+)([A-Za-z]?)', res_id_str)
        if not match:
            logger.warning(f"Could not parse residue ID '{res_id_str}' in {pdb_code}. Skipping.")
            return None
        resseq = int(match.group(1))
        icode = match.group(2)  # 插入码，可能为空字符串 ''
        # =================================================================================
        
        pdb_path = os.path.join(self.pdb_dir, f'{pdb_code}.pdb')
        if not os.path.exists(pdb_path):
            return None

        try:
            structure = prody.parsePDB(pdb_path)
            
            # 使用更精确的 prody 选择语法来定位带有插入码的残基
            sel_str = f'chain {chain_id} and resnum {resseq}'
            if icode:
                sel_str += f' and icode {icode}'
            
            mut_residue_selection = structure.select(sel_str)
            if not mut_residue_selection:
                logger.warning(f"Mutation residue not found using selector: '{sel_str}' in {pdb_code}")
                return None
            
            # 获取该残基的内部索引(resindex)
            target_resindex = mut_residue_selection.getResindices()[0]

            chain = structure.select(f'chain {chain_id} and protein')
            if not chain: return None

            resindices = sorted(list(set(chain.getResindices())))
            n_residues = len(resindices)
            pos_heavyatom = torch.zeros(n_residues, 15, 3, dtype=torch.float32)
            mask_heavyatom = torch.zeros(n_residues, 15, dtype=torch.bool)
            aa_wt = torch.zeros(n_residues, dtype=torch.long)
            
            mut_idx = -1
            
            for i, res_idx in enumerate(resindices):
                if res_idx == target_resindex:
                    mut_idx = i

                residue = chain.select(f'resindex {res_idx}')
                if residue is None: continue
                resname = residue.getResnames()[0]
                
                try:
                    aa_enum = AA(resname)
                    aa_wt[i] = aa_enum.value
                except ValueError:
                    continue

                atom_names = restype_to_heavyatom_names.get(aa_enum, [])
                for atom_idx, atom_name in enumerate(atom_names):
                    if not atom_name or atom_idx >= 15: continue
                    atom = residue.select(f'name {atom_name}')
                    if atom is not None and atom.numAtoms() > 0:
                        pos_heavyatom[i, atom_idx] = torch.from_numpy(atom.getCoords()[0])
                        mask_heavyatom[i, atom_idx] = True

            if mut_idx == -1:
                # 这个检查理论上不会失败了，但作为保险
                return None
            
            aa_mut = aa_wt.clone()
            mut_resname_char = first_mut[-1]
            from rde.utils.protein.constants import aa_short_to_long
            mut_resname_long = aa_short_to_long.get(mut_resname_char)
            if mut_resname_long:
                 aa_mut[mut_idx] = AA(mut_resname_long).value

            mut_flag = torch.zeros(n_residues, dtype=torch.bool)
            mut_flag[mut_idx] = True

            data = {
                'pdb_id': pdb_code, 'chain_id': chain_id,
                'aa_wt': aa_wt, 'aa_mut': aa_mut,
                'pos_heavyatom': pos_heavyatom,
                'mask': mask_heavyatom[:, BBHeavyAtom.CA],
                'mut_flag': mut_flag,
                'ddG': torch.tensor(row['ddG'], dtype=torch.float32)
            }
            torch.save(data, cache_path)
            return data

        except Exception as e:
            logger.error(f"Failed during processing of {pdb_code}_{chain_id}: {e}", exc_info=True)
            return None

class SKEMPIDatasetAdapter(Dataset):
    def __init__(self, base_dataset, processed_dir):
        super().__init__()
        self.base_dataset = base_dataset
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        if data is None: return None
        try:
            pdb_id, chain_id = data['pdb_id'], data['chain_id']
            id_str = f'{pdb_id}_{chain_id}'
            geo_path = os.path.join(self.processed_dir, f"{id_str}_geometry.pt")
            topo_path = os.path.join(self.processed_dir, f"{id_str}_topology.pt")
            dyn_path = os.path.join(self.processed_dir, f"{id_str}_dynamics.pt")
            if not all(os.path.exists(p) for p in [geo_path, topo_path, dyn_path]):
                return None
            data['geometric_features_wt'] = torch.load(geo_path, map_location='cpu')
            data['topological_features_wt'] = torch.load(topo_path, map_location='cpu')
            data['dynamics_features_wt'] = torch.load(dyn_path, map_location='cpu')
            data['geometric_features_mut'] = data['geometric_features_wt'].clone()
            data['topological_features_mut'] = data['topological_features_wt'].clone()
            data['dynamics_features_mut'] = data['dynamics_features_wt'].clone()
            return data
        except Exception:
            return None