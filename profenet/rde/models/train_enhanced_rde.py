#!/usr/bin/env python
"""
最终版训练脚本：从预处理数据进行高效训练
(适配GNN+Attention模型)
"""
import argparse
import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging
from datetime import datetime

# 模型和数据集的导入
from rde.models.enhanced_rde import EnhancedRDE
from rde.datasets.pdbredo_chain import PDBRedoChainDataset, PDBRedoChainAdapter
from rde.utils.transforms import get_transform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_final_rde')

def sum_weighted_losses(losses, weights):
    total_loss = 0.0
    for key, loss_val in losses.items():
        if key in weights and weights[key] > 0 and torch.is_tensor(loss_val):
            total_loss += weights[key] * loss_val
    return total_loss

def compute_loss_dict(pred_sincos, batch):
    true_chi = batch['chi']
    chi_mask = batch['chi_mask']
    res_num_chi = batch.get('res_num_chi', chi_mask.sum(dim=-1))
    true_sincos = torch.stack([torch.sin(true_chi), torch.cos(true_chi)], dim=-1)
    pred_sincos_normalized = F.normalize(pred_sincos, p=2, dim=-1)
    per_atom_loss = F.mse_loss(pred_sincos_normalized, true_sincos, reduction='none').mean(dim=-1)
    masked_loss = per_atom_loss * chi_mask
    loss_dict = {}
    for i in range(1, 5):
        res_mask = (res_num_chi == i)
        if res_mask.sum() > 0:
            denominator = chi_mask[res_mask].sum()
            loss_dict[f'mle_{i}chis'] = masked_loss[res_mask].sum() / denominator if denominator > 0 else torch.tensor(0.0, device=pred_sincos.device)
        else:
            loss_dict[f'mle_{i}chis'] = torch.tensor(0.0, device=pred_sincos.device)
    return loss_dict

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return {}
    elem = batch[0]
    result = {}
    for key in elem:
        values = [d[key] for d in batch if key in d]
        if not values: continue
        if all(isinstance(v, torch.Tensor) for v in values):
            result[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
        else:
            result[key] = values
    
    # ====[核心修正]====
    # 根据 'aa' 张量创建 padding mask
    # 假设氨基酸索引0是为padding保留的
    if 'aa' in result:
        result['mask'] = (result['aa'] != 0)
    # ===================

    return result

def parse_args():
    parser = argparse.ArgumentParser(description='Train Final RDE model')
    parser.add_argument('config', type=str, help="Path to the final config YAML file.")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_dataset(config, split):
    data_cfg = config['data']['validation'] if split == 'val' else config['data']
    transform = get_transform(data_cfg.get('transform', []))
    base_dataset = PDBRedoChainDataset(
        split=split,
        pdbredo_dir=data_cfg.get('pdbredo_dir'),
        clusters_path=data_cfg.get('clusters_path'),
        splits_path=data_cfg.get('splits_path'),
        processed_dir=data_cfg.get('processed_dir'),
        transform=transform,
        reset=False # 确保使用已有的缓存
    )
    dataset = PDBRedoChainAdapter(base_dataset)
    logger.info(f"成功加载 {split} 数据集，包含 {len(dataset)} 个样本。")
    return dataset

def train_epoch(model, dataloader, optimizer, device, epoch, config):
    model.train()
    total_loss, processed_batches = 0.0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    for batch in pbar:
        if not batch: continue
        try:
            batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_on_device)
            loss_dict = compute_loss_dict(outputs, batch_on_device)
            loss = sum_weighted_losses(loss_dict, config['training']['loss_weights'])
            if not (torch.isnan(loss) or torch.isinf(loss) or loss <= 0):
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), float(config['training']['max_grad_norm']))
                optimizer.step()
                total_loss += loss.item()
                processed_batches += 1
            if processed_batches > 0:
                pbar.set_postfix({"loss": f"{total_loss / processed_batches:.4f}"})
        except Exception as e:
            logger.error(f"训练批次中发生错误: {repr(e)}") # 使用 repr(e) 打印更详细的错误
            continue
    return total_loss / max(1, processed_batches) if processed_batches > 0 else 0.0

@torch.no_grad()
def validate(model, dataloader, device, config):
    model.eval()
    total_loss, processed_batches = 0.0, 0
    for batch in tqdm(dataloader, desc="Validation"):
        if not batch: continue
        try:
            batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_on_device)
            loss_dict = compute_loss_dict(outputs, batch_on_device)
            loss = sum_weighted_losses(loss_dict, config['training']['loss_weights'])
            if loss > 0:
                total_loss += loss.item()
                processed_batches += 1
        except Exception as e:
            logger.warning(f"验证批次中发生错误: {repr(e)}")
            continue
    return total_loss / max(1, processed_batches) if processed_batches > 0 else 0.0

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['training'].get('seed', args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['training'].get('output_dir', './outputs'), f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出将保存至: {output_dir}")
    
    train_dataset = load_dataset(config, 'train')
    val_dataset = load_dataset(config, 'val')
    
    # 增加一个检查，如果数据集为空则直接退出
    if len(train_dataset) == 0:
        logger.error("训练数据集为空，程序退出。请检查您的数据路径和文件。")
        return

    batch_size = int(config['training'].get('batch_size', 8))
    workers = min(os.cpu_count(), 12) if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=custom_collate)
    
    model = EnhancedRDE(config).to(device)
    logger.info(f"创建的GNN+Attention模型有 {sum(p.numel() for p in model.parameters()):,} 个参数。")
    
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config)
        logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}")
        
        val_loss = validate(model, val_loader, device, config)
        logger.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),}, os.path.join(output_dir, 'best_model.pt'))
            logger.info(f"已保存最佳模型，val_loss={val_loss:.6f}")
            
    logger.info("训练完成!")

if __name__ == "__main__":
    main()