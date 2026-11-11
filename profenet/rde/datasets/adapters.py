# rde/datasets/adapters.py (完整修正版)

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PDBRedoChainAdapter(torch.utils.data.Dataset):
    """
    [最终架构修正版]
    一个简单的适配器，只负责从基础数据集中获取数据，不做任何修改。
    特征生成的工作已移至主模型中。
    """
    def __init__(self, dataset, **kwargs): # 忽略所有其他参数
        self.dataset = dataset
        logger.info(f"Initialized PDBRedoChainAdapter (simple passthrough).")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 直接返回基础数据集的输出，不做任何修改
        return self.dataset[idx]


class SKEMPIDatasetAdapter(Dataset):
    def __init__(self, base_dataset, processed_dir):
        """
        初始化适配器。
        
        参数:
            base_dataset: 基础的SKEMPI数据集实例，提供原始数据。
            processed_dir: 存储预计算特征文件的目录。
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 从基础数据集中获取原始数据
        data = self.base_dataset[idx]
        if data is None:
            return None

        try:
            # 1. 为野生型(WT)加载预处理的特征
            pdb_id_wt = data['pdb_id']
            chain_id_wt = data['chain_id']
            id_str_wt = f'{pdb_id_wt}_{chain_id_wt}'

            # 构建特征文件路径
            geo_path = os.path.join(self.processed_dir, f"{id_str_wt}_geometry.pt")
            topo_path = os.path.join(self.processed_dir, f"{id_str_wt}_topology.pt")
            dyn_path = os.path.join(self.processed_dir, f"{id_str_wt}_dynamics.pt")

            # 加载特征并放入字典
            if not all(os.path.exists(p) for p in [geo_path, topo_path, dyn_path]):
                logger.warning(f"One or more feature files not found for {id_str_wt}, skipping sample.")
                return None 

            data['geometric_features_wt'] = torch.load(geo_path, map_location='cpu')
            data['topological_features_wt'] = torch.load(topo_path, map_location='cpu')
            data['dynamics_features_wt'] = torch.load(dyn_path, map_location='cpu')
            
            # 复用野生型(WT)的特征作为突变体(MUT)的特征
            data['geometric_features_mut'] = data['geometric_features_wt'].clone()
            data['topological_features_mut'] = data['topological_features_wt'].clone()
            data['dynamics_features_mut'] = data['dynamics_features_wt'].clone()

            # 重命名氨基酸序列键，以符合模型输入要求
            if 'aa' in data:
                data['aa_wt'] = data.pop('aa')

            return data

        except Exception as e:
            logger.error(f"Error loading features for index {idx} ({data.get('pdb_id', 'Unknown')}): {e}")
            return None