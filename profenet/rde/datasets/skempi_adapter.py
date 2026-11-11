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

from rde.utils.protein.constants import AA, restype_to_heavyatom_names, BBHeavyAtom

logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', PDBConstructionWarning)

class SkempiDataset(Dataset):
    """
    [最终修正版]
    这个基础类负责从原始CSV和PDB文件中解析出基本信息，
    例如坐标、氨基酸序列、突变位点和 ddG 值。
    它包含了对多链复合物ID（如 _AB_CD）的解析修正。
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
        else:
            self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = os.path.join(self.cache_dir, f'data_{idx}.pt')
        
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path)
            except Exception:
                pass

        pdb_code = row['PDB_Code']
        # [关键修正] 必须用 'Mutation(s)_cleaned' 而非 'Mutation(s)_PDB'。
        # SKEMPI v2 的 'Mutation(s)_PDB' 列所用编号与 data/SKEMPI_v2/PDBs/*.pdb 的实际
        # 残基编号不一致（例：1CSE 记为 LI45G，而该 PDB 中 I45 是 VAL；正确的是 cleaned
        # 列的 LI38G，I38 确为 LEU）。全库校验：用 _PDB 列只有 41.6% 的野生型能对上，
        # 47.5% 落在错误残基、10.9% 编号不存在；用 _cleaned 列则 100% 匹配。
        # 上游 RDE-PPI 的 rde/datasets/skempi.py:40 同样使用 _cleaned 列。
        mutations_str = row['Mutation(s)_cleaned']
        pdb_full_id = row['#Pdb']

        # [核心修正] 正确的链ID解析逻辑
        chain_groups = pdb_full_id.split('_')[1:]  # e.g., for '4P23_AB_CD', this gives ['AB', 'CD']
        all_chain_ids = [chain_id for group in chain_groups for chain_id in group] # This gives ['A', 'B', 'C', 'D']
        
        pdb_path = os.path.join(self.pdb_dir, f'{pdb_code}.pdb')
        if not os.path.exists(pdb_path):
            logger.warning(f"PDB file not found for {pdb_code}, skipping.")
            return None

        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_code, pdb_path)
            model = structure[0]

            all_residues = []
            chain_map = {chain_id: i for i, chain_id in enumerate(all_chain_ids)}
            
            for chain_id in all_chain_ids: # 使用修正后的链ID列表
                if chain_id in model:
                    all_residues.extend([res for res in model[chain_id] if res.get_id()[0] == ' '])
                else:
                    logger.warning(f"Chain {chain_id} not found in {pdb_code}, complex might be incomplete.")

            if not all_residues: return None
            
            n_residues = len(all_residues)
            #pos_heavyatom = torch.full([n_residues, 15, 3], torch.nan, dtype=torch.float32)
            pos_heavyatom = torch.zeros([n_residues, 15, 3], dtype=torch.float32)
            mask_heavyatom = torch.zeros([n_residues, 15], dtype=torch.bool)
            aa = torch.zeros(n_residues, dtype=torch.long)
            chain_nb = torch.zeros(n_residues, dtype=torch.long)
            res_nb = torch.zeros(n_residues, dtype=torch.long)
            mut_flag = torch.zeros(n_residues, dtype=torch.bool)

            for i, res in enumerate(all_residues):
                resname = res.get_resname()
                try:
                    aa[i] = AA(resname).value
                except (ValueError, KeyError):
                    aa[i] = AA.UNK.value
                    continue

                chain_nb[i] = chain_map.get(res.get_parent().id, -1) # 使用.get以增加鲁棒性
                res_nb[i] = res.get_id()[1]

                atom_names = restype_to_heavyatom_names.get(AA(resname), [])
                for atom_idx, atom_name in enumerate(atom_names):
                    if not atom_name: continue
                    if atom_name in res:
                        pos_heavyatom[i, atom_idx] = torch.from_numpy(res[atom_name].get_coord())
                        mask_heavyatom[i, atom_idx] = True
            
            aa_mut = aa.clone()
            for mut in mutations_str.split(','):
                wt_aa, chain_id, mut_aa_char = mut[0], mut[1], mut[-1]
                resseq_str = mut[2:-1]
                match = re.match(r'(-?\d+)([A-Za-z]?)', resseq_str)
                resseq, icode = int(match.group(1)), match.group(2).strip()
                
                res_id_tuple = (' ', resseq, icode if icode else ' ')
                
                found_mut = False
                for i, res in enumerate(all_residues):
                    if res.get_parent().id == chain_id and res.get_id() == res_id_tuple:
                        # [关键校验] 结构中该位点的残基必须与突变串声明的野生型一致。
                        # 否则说明编号体系不匹配（突变会被施加到错误残基上），必须丢弃而非静默污染。
                        try:
                            wt_ok = (AA(res.get_resname()) == AA(wt_aa))
                        except (ValueError, KeyError):
                            wt_ok = False
                        if not wt_ok:
                            logger.warning(
                                f"WT mismatch for {pdb_code} {mut}: structure has "
                                f"{res.get_resname()} at {chain_id}{resseq}, expected {wt_aa}")
                            return None
                        mut_flag[i] = True
                        try:
                            aa_mut[i] = AA(mut_aa_char).value
                        except (ValueError, KeyError):
                             aa_mut[i] = AA.UNK.value
                        found_mut = True
                        break
                if not found_mut:
                    logger.warning(f"Could not find residue for mutation {mut} in {pdb_code}")
                    return None

            data = {
                'id': pdb_code, 
                'chains': all_chain_ids, # 使用修正后的链ID列表
                'aa_wt': aa, 'aa_mut': aa_mut,
                'pos_heavyatom': pos_heavyatom, 'mask_heavyatom': mask_heavyatom,
                'mask': mask_heavyatom[:, BBHeavyAtom.CA],
                'chain_nb': chain_nb, 'res_nb': res_nb, 'mut_flag': mut_flag,
                'ddG': torch.tensor(row['ddG'], dtype=torch.float32),
                'metadata': {
                    'complex': row['#Pdb'],
                    'mutstr': row['Mutation(s)_cleaned'],
                }
            }
            torch.save(data, cache_path)
            return data

        except Exception as e:
            logger.error(f"Failed during BioPython processing of {pdb_code}: {e}", exc_info=False)
            return None


class SKEMPIDatasetAdapter(Dataset):
    """
    [最终版] 这个Adapter类与新的预处理脚本完全同步。
    它负责加载每个复合物对应的、预处理好的 .pt 文件，并将多条链的特征拼接起来。
    """
    def __init__(self, base_dataset, processed_dir):
        super().__init__()
        self.base_dataset = base_dataset
        self.processed_dir = processed_dir
        self.valid_indices = self._pre_filter()

    def _pre_filter(self):
        # [性能] 预过滤要遍历整个 base_dataset（每折 ~5k-7k 条），每次启动都重扫需数分钟。
        # 结果只取决于 (base_dataset 的划分, processed_dir 里有哪些特征文件)，故做磁盘缓存。
        cache_path = os.path.join(self.base_dataset.cache_dir, '_valid_indices.pt')
        sig = (len(self.base_dataset), os.path.abspath(self.processed_dir))
        if os.path.exists(cache_path):
            try:
                blob = torch.load(cache_path)
                if tuple(blob.get('sig', ())) == sig:
                    logger.info(f"SKEMPI pre-filter loaded from cache: {len(blob['valid_indices'])}/{len(self.base_dataset)}")
                    return blob['valid_indices']
            except Exception:
                pass  # 缓存损坏则重算

        valid_indices = []
        for i in tqdm(range(len(self.base_dataset)), desc="Pre-filtering SKEMPI samples"):
            data_from_base = self.base_dataset[i]
            if data_from_base is None: continue

            all_chains_exist = True
            for chain_id in data_from_base['chains']:
                pdb_code = data_from_base['id']
                pt_path = os.path.join(self.processed_dir, f"{pdb_code}_{chain_id}.pt")
                if not os.path.exists(pt_path):
                    all_chains_exist = False
                    break

            if all_chains_exist:
                valid_indices.append(i)

        logger.info(f"SKEMPI pre-filtering complete. Found {len(valid_indices)}/{len(self.base_dataset)} valid entries with all feature files.")
        try:
            torch.save({'sig': sig, 'valid_indices': valid_indices}, cache_path)
        except Exception as e:
            logger.warning(f"Could not cache pre-filter result: {e}")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        data_from_base = self.base_dataset[original_idx]
        if data_from_base is None: return None
        
        try:
            chain_features_list = []
            for chain_id in data_from_base['chains']:
                pdb_code = data_from_base['id']
                pt_path = os.path.join(self.processed_dir, f"{pdb_code}_{chain_id}.pt")
                chain_features_list.append(torch.load(pt_path))

            final_data = {}
            final_data.update(data_from_base)
            
            keys_to_concat = [
                'aa', 'pos_heavyatom', 'mask_heavyatom', 
                'phi', 'psi', 'chi', 
                'phi_mask', 'psi_mask', 'chi_mask', 
                'local_env_features'
            ]
            
            for key in keys_to_concat:
                if key in chain_features_list[0]:
                    final_data[key] = torch.cat([cf[key] for cf in chain_features_list], dim=0)

            if 'mask_heavyatom' in final_data:
                final_data['mask'] = final_data['mask_heavyatom'][:, BBHeavyAtom.CA]

            return final_data
        except Exception as e:
            logger.error(f"Error loading/concatenating .pt files for index {idx} (original {original_idx}): {e}")
            return None
