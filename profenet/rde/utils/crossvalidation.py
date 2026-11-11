import torch
import torch.optim as optim

class CrossValidation:
    """管理多个模型实例用于交叉验证"""
    def __init__(self, model_factory, config, num_cvfolds, use_geometric=None, use_topological=None, device='cuda'):
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.device = device
        
        # 如果未指定，从配置中读取几何和拓扑特征选项
        if use_geometric is None:
            use_geometric = getattr(config.model, 'use_geometric', True)
        if use_topological is None:
            use_topological = getattr(config.model, 'use_topological', True)
        
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        # 为每个折创建模型、优化器和调度器
        for fold in range(num_cvfolds):
            # 创建模型，传递几何和拓扑特征选项
            model = model_factory(config.model, use_geometric=use_geometric, use_topological=use_topological)
            model = model.to(device)
            
            # 处理不同的优化器配置格式
            if hasattr(config.train, 'optimizer') and hasattr(config.train.optimizer, 'lr'):
                # 使用原始SKEMPI格式
                lr = float(config.train.optimizer.lr)
                weight_decay = float(config.train.optimizer.get('weight_decay', 0.0))
                beta1 = float(config.train.optimizer.get('beta1', 0.9))
                beta2 = float(config.train.optimizer.get('beta2', 0.999))
                
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(beta1, beta2)
                )
            else:
                # 使用新格式
                lr = float(config.train.get('learning_rate', 0.0005))
                weight_decay = float(config.train.get('weight_decay', 0.0001))
                
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            
            # 处理不同的调度器配置格式
            if config.train.scheduler.type == 'plateau':
                # 获取参数，确保向后兼容
                factor = float(config.train.scheduler.get('factor', 0.5))
                patience = int(config.train.scheduler.get('patience', 10))
                min_lr = float(config.train.scheduler.get('min_lr', 0.00001))
                
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr
                )
            else:
                # 使用StepLR作为备选
                step_size = int(config.train.scheduler.get('step_size', 5000))
                gamma = float(config.train.scheduler.get('gamma', 0.5))
                
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
            
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
    
    def get(self, fold):
        """获取指定折的模型、优化器和调度器"""
        return self.models[fold], self.optimizers[fold], self.schedulers[fold]
    
    def state_dict(self):
        """获取所有模型和优化器的状态"""
        return {
            'models': [model.state_dict() for model in self.models],
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'schedulers': [scheduler.state_dict() for scheduler in self.schedulers]
        }
    
    def load_state_dict(self, state_dict):
        """加载所有模型和优化器的状态"""
        for i in range(self.num_cvfolds):
            self.models[i].load_state_dict(state_dict['models'][i])
            self.optimizers[i].load_state_dict(state_dict['optimizers'][i])
            self.schedulers[i].load_state_dict(state_dict['schedulers'][i])
    
    def to(self, device):
        """将所有模型移到指定设备"""
        for model in self.models:
            model.to(device)
        return self