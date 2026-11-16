import torch
import torch.nn as nn
import torch.nn.functional as F

# === 基础组件 ===
class FeatureProjector(nn.Module):
    """
    特征投影：Raw Dim -> Latent Dim
    使用 LayerNorm + Dropout 增强鲁棒性
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.norm(self.proj(x)))

# === 1. 跨语句因果因子提取器 ===
class CausalFactorExtractor(nn.Module):
    """
    输入：当前语句序列 (Batch, Seq_Len, Dim)
    输出：包含因果信息的上下文特征 Phi_causal
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, 200, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 使用 Transformer 提取时序依赖
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, 
            dropout=0.3, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x_curr, x_ctx=None):
        h_full = x_curr 
            
        B, L, _ = h_full.shape
        
        # Positional Encoding
        if L <= 200:
            h_full = h_full + self.pos_enc[:, :L, :]
        else:
            h_full = h_full + self.pos_enc[:, :200, :] # 简单截断处理防止crash
        
        # Causal Mask (只能看过去)
        mask = torch.triu(torch.ones(L, L, device=x_curr.device) * float('-inf'), diagonal=1)
        
        # Transformer Encoding
        features = self.transformer(h_full, mask=mask)
        
        return self.norm(features)

# === 2. Mixture of Experts (MoE) 生成器 ===
class MoEBlock(nn.Module):
    """
    MoE 核心模块：Gate + Experts
    """
    def __init__(self, hidden_dim, num_experts=4, k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        # [FIX] 自动调整 k 值，防止 k > num_experts 导致的崩溃
        self.k = min(k, num_experts) 
        
        # 专家网络 (简单的 MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # 1. 计算 Gate Logits
        gate_logits = self.gate(x) # [B, S, NumExperts]
        
        # 2. Top-K Routing
        weights, indices = torch.topk(F.softmax(gate_logits, dim=-1), self.k, dim=-1)
        
        # 3. 计算 Expert 输出
        batch_size, seq_len, dim = x.shape
        final_output = torch.zeros_like(x)
        
        # 归一化权重
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        for i in range(self.k):
            expert_idx = indices[:, :, i] # [B, S]
            weight = weights[:, :, i].unsqueeze(-1) # [B, S, 1]
            
            expert_outputs = torch.stack([e(x) for e in self.experts], dim=2) # [B, S, NumExperts, D]
            
            selected_output = torch.gather(
                expert_outputs, 
                2, 
                expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, dim)
            ).squeeze(2)
            
            final_output += weight * selected_output
            
        return final_output

class CausalMoEGen(nn.Module):
    """
    完整生成器模块 - 支持消融实验 (use_causal=False)
    """
    def __init__(self, latent_dim, num_experts=4, use_causal=True):
        super().__init__()
        self.use_causal = use_causal 
        
        if self.use_causal:
            self.extractor = CausalFactorExtractor(latent_dim, latent_dim)
        
        self.moe_layers = nn.ModuleList([
            # 这里即使 num_experts=1，传入 MoEBlock 后 k 会自动修正为 1
            MoEBlock(latent_dim, num_experts=num_experts) for _ in range(2)
        ])
        
        self.final_proj = nn.Linear(latent_dim, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x_curr, x_ctx=None):
        # 1. 提取因果上下文 (如果启用)
        if self.use_causal:
            h = self.extractor(x_curr, x_ctx)
        else:
            # [Ablation] 不使用因果提取，直接使用输入的局部特征
            h = x_curr 
        
        # 2. MoE 生成 (Residual)
        for layer in self.moe_layers:
            h = h + layer(h)
            
        return self.out_norm(self.final_proj(h))

# === 3. 置信度门控模块 ===
class ConfidenceGating(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim), 
            nn.Sigmoid()
        )
        # 初始化 Bias 为负数，使初始 Gate 值接近 0
        nn.init.constant_(self.gate_net[-2].bias, -2.0)
    
    def forward(self, gen_x):
        # 计算置信度 (0~1)
        confidence = self.gate_net(gen_x)
        return gen_x * confidence, confidence