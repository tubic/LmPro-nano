# rde/datasets/pdbredo_chain.py (最终修复版 - 修复所有 NameError)

import os
import math
import random
import pickle
import collections
import torch
import lmdb
import logging
import numpy as np
import traceback
from typing import Mapping, List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from rde.utils.protein.parsers import parse_biopython_structure

logger = logging.getLogger(__name__)

def _process_structure(cif_path, structure_id) -> Optional[Dict]:
    parser = MMCIFParser(QUIET=True)
    try:
        model = parser.get_structure(structure_id, cif_path)[0]
    except (PDBConstructionException, FileNotFoundError, IndexError):
        logger.warning(f'[预处理] 加载或解析结构失败: {cif_path}。')
        return None
    data, _ = parse_biopython_structure(model)
    if data is None:
        logger.warning(f'[预处理] 解析结构后有效残基过少: {cif_path}')
        return None
    data['id'] = structure_id
    return data

class PDBRedoChainDataset(Dataset):
    MAP_SIZE = 384*(1024*1024*1024)
    MIN_CHAIN_LENGTH = 20
    _LMDB_ENVS: Dict[Tuple[int, str], object] = {}

    def __init__(
        self,
        split,
        pdbredo_dir = './data/PDB_REDO',
        clusters_path = './data/pdbredo_clusters.txt',
        splits_path = './data/pdbredo_splits.txt',
        processed_dir = './data/PDB_REDO_processed',
        num_preprocess_jobs = math.floor(cpu_count() * 0.8),
        transform = None,
        sample_by_cluster = False,
        cache_namespace = None,
        structure_cache_dir = None,
        reset = False,
    ):
        super().__init__()
        self.split = split
        self.pdbredo_dir = pdbredo_dir
        self.clusters_path = clusters_path
        self.splits_path = splits_path
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        self.num_preprocess_jobs = num_preprocess_jobs
        self.transform = transform
        self.sample_by_cluster = sample_by_cluster
        self.cache_namespace = cache_namespace
        # Keep large immutable LMDB reads on a local disk when available while
        # retaining split-specific feature/index files in processed_dir.
        self.structure_cache_dir = structure_cache_dir or processed_dir
        self.clusters: Mapping[str, List[Tuple[str, str]]] = collections.defaultdict(list)
        self.splits: Mapping[str, List[str]] = collections.defaultdict(list)
        self._load_clusters()
        self._load_splits()
        self.db_conn = None
        self._db_pid = None
        self.db_keys: Optional[List[str]] = None
        self._preprocess_structures(reset)
        self._sanitize_clusters(reset)
        clusters_of_split = [c for c in self.splits.get(split, []) if c in self.clusters]
        self.flat_chain_list: List[Tuple[str, str]] = []
        for clust_name in clusters_of_split:
            for pdb_chain_tuple in self.clusters[clust_name]:
                self.flat_chain_list.append(pdb_chain_tuple)
        self._clusters_of_split = clusters_of_split
        if self.sample_by_cluster:
            logger.info(
                f"数据集 {split} 按序列簇采样，包含 "
                f"{len(self._clusters_of_split)} 个簇。"
            )
        else:
            logger.info(f"数据集 {split} 已被展开，包含 {len(self.flat_chain_list)} 个独立的蛋白质链。")

    @property
    def lmdb_path(self): return os.path.join(self.structure_cache_dir, 'structures.lmdb')
    @property
    def keys_path(self): return os.path.join(self.structure_cache_dir, 'keys.pkl')
    @property
    def sanitized_clusters_path(self):
        suffix = f'_{self.cache_namespace}' if self.cache_namespace else ''
        return os.path.join(self.processed_dir, f'sanitized_clusters{suffix}.pkl')
    def _load_clusters(self):
        with open(self.clusters_path, 'r') as f: lines = f.readlines()
        current_cluster = None
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith('[') and line.endswith(']'): current_cluster = line[1:-1]
            elif current_cluster is not None:
                for word in line.split():
                    try: pdbcode, chain_id = word.split(':'); self.clusters[current_cluster].append( (pdbcode, chain_id) )
                    except ValueError: logging.warning(f"跳过簇 {current_cluster} 中的格式错误条目: {word}"); continue
    def _load_splits(self):
        with open(self.splits_path, 'r') as f: lines = f.readlines()
        current_split = None
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith('[') and line.endswith(']'): current_split = line[1:-1]
            elif current_split is not None and line: self.splits[current_split].extend(line.split())
    def get_all_pdbcodes(self):
        pdbcodes = set(); [pdbcodes.add(pdbcode) for _, pdbchain_list in self.clusters.items() for pdbcode, _ in pdbchain_list]; return pdbcodes
    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path) and not reset: return
        pdbcodes = self.get_all_pdbcodes(); tasks = []
        for pdbcode in pdbcodes:
            cif_path = os.path.join(self.pdbredo_dir, pdbcode[1:3].lower(), pdbcode, f"{pdbcode}_final.cif")
            if not os.path.exists(cif_path): continue
            tasks.append(delayed(_process_structure)(cif_path, pdbcode))
        db_conn = lmdb.open(self.lmdb_path, map_size=self.MAP_SIZE, create=True, subdir=False, readonly=False)
        keys = []
        with db_conn.begin(write=True) as txn:
            processed_data = Parallel(n_jobs=self.num_preprocess_jobs)(tqdm(tasks, desc='Preprocessing structures'))
            for data in processed_data:
                if data is None: continue
                key = data['id']; keys.append(key); txn.put(key=key.encode(), value=pickle.dumps(data))
        db_conn.close()
        with open(self.keys_path, 'wb') as f: pickle.dump(keys, f)
    def _connect_db(self):
        pid = os.getpid()
        if self.db_conn is not None and self._db_pid == pid:
            return

        # LMDB permits one environment per path in a process.  Train and
        # validation datasets use the same cache, while worker processes must
        # obtain their own handle after a DataLoader fork.
        self.db_conn = None
        absolute_path = os.path.abspath(self.lmdb_path)
        # After a fork, the child inherits the parent's Environment object.
        # LMDB refuses a second open for the same path until that inherited
        # handle is closed. Closing it here affects only the child's memory.
        for cached_key, environment in list(self._LMDB_ENVS.items()):
            cached_pid, cached_path = cached_key
            if cached_path == absolute_path and cached_pid != pid:
                environment.close()
                del self._LMDB_ENVS[cached_key]

        env_key = (pid, absolute_path)
        if env_key not in self._LMDB_ENVS:
            self._LMDB_ENVS[env_key] = lmdb.open(
                self.lmdb_path,
                map_size=self.MAP_SIZE,
                create=False,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        self.db_conn = self._LMDB_ENVS[env_key]
        self._db_pid = pid
        with open(self.keys_path, 'rb') as f: self.db_keys = pickle.load(f)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['db_conn'] = None
        state['_db_pid'] = None
        return state
    def _get_from_db(self, pdbcode):
        self._connect_db()
        data_bytes = self.db_conn.begin().get(pdbcode.encode())
        if data_bytes is None: raise KeyError(f"PDB code {pdbcode} not found in LMDB.")
        return pickle.loads(data_bytes)
    def _sanitize_clusters(self, reset):
        if os.path.exists(self.sanitized_clusters_path) and not reset:
            with open(self.sanitized_clusters_path, 'rb') as f: self.clusters = pickle.load(f); return
        self._connect_db(); db_keys_set = set(self.db_keys); clusters_sanitized = collections.defaultdict(list)
        for clust_name, pdbchain_list in tqdm(self.clusters.items(), desc='Sanitizing clusters'):
            sanitized_list = []
            for pdbcode, chain_id in pdbchain_list:
                if pdbcode in db_keys_set:
                    data = self._get_from_db(pdbcode)
                    if 'chain_id' in data and chain_id in data['chain_id']: sanitized_list.append((pdbcode, chain_id))
            if sanitized_list: clusters_sanitized[clust_name] = sanitized_list
        with open(self.sanitized_clusters_path, 'wb') as f: pickle.dump(clusters_sanitized, f)
        self.clusters = clusters_sanitized
    def __len__(self):
        if self.sample_by_cluster:
            return len(self._clusters_of_split)
        return len(self.flat_chain_list)

    def _get_cluster_sample(self, index):
        """Match the published RDE-PPI pretraining sampler exactly.

        One structure-chain pair is sampled from each 50%-identity cluster on
        every visit. The full structure is retained so the focused patch can
        include spatial neighbours outside the selected chain.
        """
        if isinstance(index, int):
            index = (index, None)
        cluster_index, member_index = index
        if cluster_index >= len(self._clusters_of_split):
            raise IndexError("Index out of range.")

        cluster = self._clusters_of_split[cluster_index]
        members = self.clusters[cluster]
        if member_index is None:
            pdbcode, chain = random.choice(members)
        else:
            pdbcode, chain = members[member_index]

        data = self._get_from_db(pdbcode)
        focus_flag = torch.tensor(
            [chain_id == chain for chain_id in data['chain_id']],
            dtype=torch.bool,
        )
        if not focus_flag.any():
            raise KeyError(f"Chain {chain} not found in PDB code {pdbcode}.")
        data['focus_flag'] = focus_flag
        data['focus_chain'] = chain
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getitem__(self, index):
        if self.sample_by_cluster:
            return self._get_cluster_sample(index)
        try:
            if index >= len(self.flat_chain_list): raise IndexError("Index out of range.")
            pdbcode, chain = self.flat_chain_list[index]
            data_all_chains = self._get_from_db(pdbcode).copy()
            
            # 【修复】使用 np.array 替换 numpy.array
            chain_mask = (np.array(data_all_chains['chain_id']) == chain)
            if not chain_mask.any():
                logger.warning(f"[数据集] 在PDB '{pdbcode}' 中找不到链 '{chain}'，跳过索引 {index}。"); return None
            
            # 【修复】使用 np.array 替换 numpy.array
            aa_seq = np.array(data_all_chains['aa'])[chain_mask]
            if len(aa_seq) < self.MIN_CHAIN_LENGTH:
                logger.warning(f"[数据集] 样本 {pdbcode}_{chain} 链长({len(aa_seq)})过短，小于{self.MIN_CHAIN_LENGTH}，跳过索引 {index}。"); return None
            
            data = {'id': f"{pdbcode}_{chain}"}
            numerical_keys = ['aa', 'chain_nb', 'resseq', 'res_nb', 'pos_heavyatom', 'mask_heavyatom', 'bfactor_heavyatom', 'phi', 'phi_mask', 'psi', 'psi_mask', 'chi', 'chi_alt', 'chi_mask', 'chi_complete']
            string_keys = ['chain_id', 'icode']
            for key in numerical_keys:
                if key in data_all_chains:
                    # 【修复】使用 np.array 替换 numpy.array
                    data[key] = torch.as_tensor(np.array(data_all_chains[key])[chain_mask])
            for key in string_keys:
                if key in data_all_chains:
                    # 【修复】使用 np.array 替换 numpy.array
                    data[key] = np.array(data_all_chains[key])[chain_mask].tolist()
            return data
        except Exception as e:
            logger.error(f"[数据集] 处理索引 {index} 时发生未知严重错误: {e}"); logger.error(traceback.format_exc()); return None

class PDBRedoChainAdapter(Dataset):
    def __init__(self, base_dataset, transform=None, load_precomputed_features=True):
        super().__init__()
        self.base_dataset = base_dataset
        self.processed_dir = self.base_dataset.processed_dir
        self.transform = transform
        self.load_precomputed_features = bool(load_precomputed_features)

        namespace = self.base_dataset.cache_namespace
        suffix = f'_{namespace}' if namespace else ''
        cache_path = os.path.join(
            self.processed_dir,
            f"valid_indices_{self.base_dataset.split}{suffix}.pkl",
        )
        if os.path.exists(cache_path):
            logger.info(f"正在从缓存加载已筛选的索引: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.valid_indices = pickle.load(f)
        else:
            logger.info(f"未找到缓存文件，正在运行预筛选...")
            self.valid_indices = self._pre_filter()
            logger.info(f"筛选完成，正在将索引缓存至: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.valid_indices, f)

    def _pre_filter(self):
        logger.info(f"开始使用[多模态特征检查]方法预筛选数据集 {self.base_dataset.split}...")
        valid_indices = []
        for i in tqdm(range(len(self.base_dataset)), desc=f"预筛选 {self.base_dataset.split} 数据集"):
            try:
                sample_preview = self.base_dataset[i]
                if sample_preview is None: continue
                if 'chi' not in sample_preview or 'chi_mask' not in sample_preview: continue
                pdb_id = sample_preview.get('id')
                if not pdb_id: continue

                if self.load_precomputed_features:
                    self._attach_precomputed_features(sample_preview, pdb_id)
                if self.transform:
                    transformed_sample = self.transform(sample_preview)
                    if transformed_sample is not None and 'chi_native' in transformed_sample and 'chi_corrupt' in transformed_sample:
                        valid_indices.append(i)
                else:
                    valid_indices.append(i)
            except Exception:
                continue
        
        original_size = len(self.base_dataset)
        final_size = len(valid_indices)
        logger.info(f"预筛选完成。有效样本数: {final_size}/{original_size} (移除了 {original_size - final_size} 个样本)")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def _attach_precomputed_features(self, sample, pdb_id):
        feature_paths = {
            'geometry_features': os.path.join(
                self.processed_dir, f"{pdb_id}_geometry.npy"
            ),
            'dihedral_features': os.path.join(
                self.processed_dir, f"{pdb_id}_dihedral.npy"
            ),
            'local_env_features': os.path.join(
                self.processed_dir, f"{pdb_id}_local_env.npy"
            ),
        }
        missing = [path for path in feature_paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing precomputed features for {pdb_id}: {missing}"
            )
        for key, path in feature_paths.items():
            sample[key] = torch.from_numpy(np.load(path))

    def __getitem__(self, index):
        base_index = self.valid_indices[index]
        try:
            sample = self.base_dataset[base_index]
            if sample is None: return None
            pdb_id = sample.get('id', f'Unknown_ID_at_index_{base_index}')

            if self.load_precomputed_features:
                self._attach_precomputed_features(sample, pdb_id)
            # 兼容 patch 需要的键名（重原子坐标）
            if 'pos_heavyatom' in sample and 'mask_heavyatom' in sample:
                sample['pos_atoms']  = sample['pos_heavyatom']
                sample['mask_atoms'] = sample['mask_heavyatom']

            if self.transform is not None:
                sample = self.transform(sample)
            return sample
        except Exception as e:
            logger.error(f"[Adapter] 处理有效索引 {index} (原始索引 {base_index}) 时发生意外错误: {e}")
            logger.error(traceback.format_exc())
            return None


def get_pdbredo_chain_dataset(cfg):
    """Build the cluster-sampled dataset used by the official RDE trainer."""
    from rde.utils.transforms import get_transform

    return PDBRedoChainDataset(
        split=cfg.split,
        pdbredo_dir=cfg.pdbredo_dir,
        clusters_path=cfg.clusters_path,
        splits_path=cfg.splits_path,
        processed_dir=cfg.processed_dir,
        transform=get_transform(cfg.transform),
        sample_by_cluster=True,
    )
