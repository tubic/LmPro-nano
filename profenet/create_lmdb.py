# create_lmdb.py

import os
import glob
import lmdb
import pickle
import torch
import argparse
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('create_lmdb')

def create_lmdb_from_pt_files(pt_files_dir: str, lmdb_path: str):
    """
    从 .pt 文件目录创建一个 LMDB 数据库。
    """
    # 查找所有 .pt 文件
    pt_files = glob.glob(os.path.join(pt_files_dir, '*.pt'))
    if not pt_files:
        logger.error(f"在目录 '{pt_files_dir}' 中未找到任何 .pt 文件！")
        return
    
    logger.info(f"找到了 {len(pt_files)} 个 .pt 文件，将把它们写入到 '{lmdb_path}'")
    
    # 设置一个足够大的map_size，例如 1TB
    # 这不是实际占用的硬盘空间，而是数据库的最大理论容量
    map_size = 1024 * 1024 * 1024 * 1024  # 1TB

    # 创建LMDB环境
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    # 开始写入事务
    with env.begin(write=True) as txn:
        for i, pt_file in enumerate(tqdm(pt_files, desc="Writing to LMDB")):
            try:
                # 加载 .pt 文件
                sample = torch.load(pt_file, map_location='cpu')
                if sample is None:
                    logger.warning(f"文件 '{pt_file}' 内容为空，已跳过。")
                    continue
                
                # 使用整数索引作为键
                key = str(i).encode('utf-8')
                # 使用 pickle 序列化样本字典
                value = pickle.dumps(sample)
                
                # 存入数据库
                txn.put(key, value)
            except Exception as e:
                logger.error(f"处理文件 '{pt_file}' 时出错: {e}", exc_info=True)
                continue

    logger.info(f"LMDB 数据库创建成功，总共写入了约 {i+1} 条记录。")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="将预处理后的 .pt 文件打包成一个高效的 LMDB 数据库。")
    parser.add_argument('pt_dir', type=str, help="存放 .pt 文件的目录 (例如 './data/PDB_REDO_preprocessed_final/train')。")
    parser.add_argument('lmdb_output_path', type=str, help="生成的 .lmdb 文件的完整路径 (例如 './data/train.lmdb')。")
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.lmdb_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    create_lmdb_from_pt_files(args.pt_dir, args.lmdb_output_path)

if __name__ == '__main__':
    main()
