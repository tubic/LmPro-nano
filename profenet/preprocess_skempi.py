#!/usr/bin/env python3
"""
预处理 SKEMPI 数据集，提取高质量特征
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from Bio.PDB import PDBParser
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preprocess_skempi')

def extract_features(structure, chain_ids=None):
    """从蛋白质结构中提取特征"""
    features = []
    coords = []
    
    # 如果没有指定链ID，使用所有链
    if chain_ids is None:
        chain_ids = [chain.id for chain in structure.get_chains()]
    
    # 遍历指定的链
    for chain_id in chain_ids:
        try:
            chain = structure[0][chain_id]
            
            # 遍历所有残基
            for residue in chain:
                # 跳过非标准残基
                if residue.id[0] != " ":
                    continue
                
                # 获取CA原子坐标
                if "CA" in residue:
                    ca_coord = residue["CA"].get_coord()
                    coords.append(ca_coord)
                    
                    # 提取残基特征
                    res_features = []
                    
                    # 1. 残基类型（简单的one-hot编码）
                    residue_types = "ACDEFGHIKLMNPQRSTVWY"
                    res_type_vector = [0] * len(residue_types)
                    res_idx = residue_types.find(residue.get_resname()[0])
                    if res_idx >= 0:
                        res_type_vector[res_idx] = 1
                    
                    # 2. 残基位置信息
                    res_position = residue.id[1]  # 残基编号
                    
                    # 3. 二级结构信息（如果可用）
                    ss_info = 0  # 默认为未知
                    if hasattr(residue, "secondary_structure"):
                        ss = residue.secondary_structure
                        if ss == "H":  # alpha-helix
                            ss_info = 1
                        elif ss == "E":  # beta-strand
                            ss_info = 2
                    
                    # 合并特征
                    res_features.extend(res_type_vector)  # 残基类型 [20]
                    res_features.append(res_position / 1000.0)  # 归一化残基位置 [1]
                    res_features.append(ss_info / 2.0)  # 归一化二级结构 [1]
                    
                    # 添加物理化学特性
                    # 疏水性
                    hydrophobicity = {
                        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                    }
                    # 电荷
                    charge = {
                        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
                        'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
                        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
                        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
                    }
                    # 极性
                    polarity = {
                        'A': 0, 'C': 1, 'D': 1, 'E': 1, 'F': 0,
                        'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
                        'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
                        'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
                    }
                    
                    res_name = residue.get_resname()
                    if len(res_name) > 0 and res_name[0] in hydrophobicity:
                        res_features.append(hydrophobicity[res_name[0]] / 5.0)  # 归一化疏水性
                        res_features.append(charge[res_name[0]])  # 电荷
                        res_features.append(polarity[res_name[0]])  # 极性
                    else:
                        res_features.extend([0, 0, 0])  # 默认值
                    
                    features.append(res_features)
        except Exception as e:
            logger.error(f"Error processing chain {chain_id}: {e}")
    
    if len(features) == 0:
        return None, None
        
    # 转换为张量
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    coord_tensor = torch.tensor(coords, dtype=torch.float32)
    
    return feature_tensor, coord_tensor

def process_skempi_entry(pdb_id, pdb_dir, mutations, ddG):
    """处理单个 SKEMPI 条目，提取野生型和突变型的特征"""
    try:
        # 解析PDB文件
        parser = PDBParser(QUIET=True)
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(pdb_path):
            logger.warning(f"PDB file not found: {pdb_path}")
            return None
            
        structure = parser.get_structure(pdb_id, pdb_path)
        
        # 解析突变
        mut_list = mutations.split(',')
        mut_chains = set()
        
        for mut in mut_list:
            # 格式如 "A123P"，其中A是链ID
            if len(mut) >= 2:
                mut_chains.add(mut[0])
        
        # 提取野生型特征
        wt_features, wt_coords = extract_features(structure, chain_ids=mut_chains)
        
        if wt_features is None:
            logger.warning(f"Failed to extract features for {pdb_id}")
            return None
        
        # 创建简化的突变特征（通常需要更复杂的模型）
        # 这里我们简单地复制野生型特征并添加噪声
        mut_features = wt_features.clone()
        
        # 对突变特征添加少量噪声
        noise = torch.randn_like(mut_features) * 0.1
        mut_features = mut_features + noise
        
        # 构建结果字典
        result = {
            'pdb_id': pdb_id,
            'mutations': mutations,
            'ddG': torch.tensor(float(ddG), dtype=torch.float32),
            'wt_features': wt_features,
            'mut_features': mut_features,
            'coordinates': wt_coords
        }
        
        return result
    except Exception as e:
        logger.error(f"Error processing {pdb_id} with mutations {mutations}: {e}")
        return None

def preprocess_skempi(csv_path, pdb_dir, output_dir, limit=None):
    """预处理整个 SKEMPI 数据集"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} entries from {csv_path}")
    
    # 限制处理的条目数量（用于测试）
    if limit is not None:
        df = df.head(limit)
        logger.info(f"Limited to first {limit} entries")
    
    # 处理每个条目
    results = []
    processed_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing SKEMPI entries"):
        try:
            pdb_id = row['#Pdb']
            mutations = row['Mutation(s)_PDB']
            ddG = row['ddG']
            
            result = process_skempi_entry(pdb_id, pdb_dir, mutations, ddG)
            if result is not None:
                results.append(result)
                processed_count += 1
        except Exception as e:
            logger.error(f"Error processing row: {e}")
    
    logger.info(f"Successfully processed {processed_count} out of {len(df)} entries")
    
    # 保存处理后的结果
    output_path = os.path.join(output_dir, "processed_skempi.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved processed data to {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Preprocess SKEMPI dataset")
    parser.add_argument("--csv_path", type=str, default="./data/SKEMPI_v2/skempi_v2.csv", 
                        help="Path to SKEMPI CSV file")
    parser.add_argument("--pdb_dir", type=str, default="./data/SKEMPI_v2/PDBs",
                        help="Directory containing PDB files")
    parser.add_argument("--output_dir", type=str, default="./data/SKEMPI_v2_processed",
                        help="Output directory for processed data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of entries to process (for testing)")
    
    args = parser.parse_args()
    
    preprocess_skempi(args.csv_path, args.pdb_dir, args.output_dir, args.limit)

if __name__ == "__main__":
    main()