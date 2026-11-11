# fast_dataset_lmdb.py

import lmdb
import pickle
from torch.utils.data import Dataset
import logging
import os
logger = logging.getLogger(__name__)

class LMDBDataset(Dataset):
    """
    从 LMDB 数据库中高效读取预处理好的样本。
    这是性能最高的数据加载方式。
    """
    def __init__(self, lmdb_path: str):
        """
        初始化数据集。
        :param lmdb_path: 指向 .lmdb 文件的路径。
        """
        super().__init__()
        self.lmdb_path = lmdb_path
        
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(
                f"错误：LMDB 文件未找到: '{lmdb_path}'\n"
                f"请您先运行 `python create_lmdb.py` 脚本来生成数据库文件。"
            )

        # 以只读模式连接到数据库，lock=False对多进程读取更友好
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        # 获取数据集的总长度
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            
        logger.info(f"成功连接到 LMDB 数据库 '{lmdb_path}'. 数据集长度: {self.length}.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        """
        通过索引从 LMDB 中获取、反序列化并返回一个样本。
        """
        if index >= self.length:
            raise IndexError("Index out of range")
            
        try:
            # 在只读事务中获取数据
            with self.env.begin(write=False) as txn:
                key = str(index).encode('utf-8')
                value_bytes = txn.get(key)
            
            if value_bytes is None:
                logger.error(f"在LMDB中未找到键: {index}")
                return None
                
            # 使用 pickle 反序列化
            sample = pickle.loads(value_bytes)
            return sample
        except Exception as e:
            logger.error(f"从LMDB加载索引 {index} 时出错: {e}", exc_info=True)
            return None
