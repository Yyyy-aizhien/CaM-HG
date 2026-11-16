import torch

class Config:
    def __init__(self):
        self.model_name = 'Causal_OT_HgVT_SOTA_Final'
        self.dataset_name = 'IEMOCAP'
        self.seed = 1111
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = './checkpoints'
        
        # === 训练基础配置 (SDR-GNN 对齐) ===
        self.batch_size = 32
        self.num_classes = 6
        self.epochs = 100
        self.lr = 1e-4
        self.weight_decay = 1e-3 # 稍微加大一点，配合 Dropout 0.5
        self.grad_clip = 5.0
        self.early_stop = 40
        
        # === [策略 3] SDR 风格 Mask ===
        # 训练时只遮 10%，让模型先"学会"，而不是"猜对"
        self.mask_ratio = 0.1  
        
        # === [策略 1] 增大模型容量 (Scale Up) ===
        self.hidden_dim = 512  # 回到 512
        self.dropout = 0.5     # 强正则化，防止 512 维过拟合
        self.num_layers = 2    # 保持 2 层即可，太深难训
        
        # === Loss 权重 ===
        self.weight_fm = 1.0
        self.weight_ce = 1.0
        self.weight_rec = 2.0 
        self.weight_pop = 0.01
        self.weight_div = 0.01

opt = Config()