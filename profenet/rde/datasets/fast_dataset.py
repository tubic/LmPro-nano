# fast_dataset.py

import os
import torch
import glob
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class FastPreprocessedDataset(Dataset):
    """
    一个高效的数据集类，用于读取由 preprocess.py 生成的 .pt 文件。
    这个类的 __getitem__ 方法非常快，因为它只涉及单个文件读取操作。
    """
    def __init__(self, processed_dir: str, split: str):
        """
        初始化数据集。
        :param processed_dir: 包含 'train' 和 'val' 子目录的根文件夹路径。
        :param split: 'train' 或 'val'。
        """
        super().__init__()
        self.data_dir = os.path.join(processed_dir, split)
        
        # 使用 glob 快速查找所有 .pt 文件
        self.file_paths = glob.glob(os.path.join(self.data_dir, '*.pt'))
        
        if not self.file_paths:
            raise RuntimeError(
                f"错误：在目录 '{self.data_dir}' 中未找到任何预处理的 .pt 文件！\n"
                f"请您先运行 `python preprocess.py <您的配置文件>` 来生成这些文件。"
            )
            
        logger.info(f"成功初始化 FastPreprocessedDataset for '{split}' split. 发现 {len(self.file_paths)} 个预处理样本。")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> dict:
        """
        加载并返回一个预处理好的样本。
        这个操作非常快，是性能的关键。
        """
        filepath = self.file_paths[index]
        try:
            # 这是此函数中唯一的操作，非常高效！
            data = torch.load(filepath, map_location='cpu')
            return data
        except Exception as e:
            logger.error(f"加载文件 '{filepath}' 时出错: {e}", exc_info=True)
            # 返回一个空字典或None，让collate_fn可以处理它
            return None
