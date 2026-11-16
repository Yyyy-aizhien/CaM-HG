import torch
import torch.nn as nn
import torch.nn.functional as F
from hypergraph_layers import CausalDynamicAdjacency, CausalHypergraphAttention, FeedForward, DropPath

class AsymmetricCausalHypergraph(nn.Module):
    def __init__(self, adim, tdim, vdim, hidden_dim=256, num_layers=2, num_hyperedges=16, num_speakers=4, drop_path=0.1):
        super().__init__()
        self.adim = adim; self.tdim = tdim; self.vdim = vdim; self.hidden_dim = hidden_dim
        
        if adim > 0: self.proj_a = nn.Sequential(nn.Conv1d(adim, hidden_dim, 3, padding=1), nn.BatchNorm1d(hidden_dim), nn.GELU())
        else: self.proj_a = nn.Identity()
        if tdim > 0: self.proj_t = nn.Sequential(nn.Conv1d(tdim, hidden_dim, 3, padding=1), nn.BatchNorm1d(hidden_dim), nn.GELU())
        else: self.proj_t = nn.Identity()
        if vdim > 0: self.proj_v = nn.Sequential(nn.Conv1d(vdim, hidden_dim, 3, padding=1), nn.BatchNorm1d(hidden_dim), nn.GELU())
        else: self.proj_v = nn.Identity()
        
        self.speaker_emb = nn.Embedding(num_speakers, hidden_dim)
        self.modality_emb = nn.Parameter(torch.randn(1, 1, 3, hidden_dim) * 0.02)
        self.edge_init = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        
        self.layers = nn.ModuleList()
        # [TRICK] DropPath Rate increasing layer by layer
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)] 
        
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'adj': CausalDynamicAdjacency(hidden_dim),
                'agg_attn': CausalHypergraphAttention(hidden_dim),
                'dist_attn': CausalHypergraphAttention(hidden_dim),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'ffn_e': FeedForward(hidden_dim),
                'ffn_v': FeedForward(hidden_dim),
                'drop_path': DropPath(dpr[i]) # [NEW]
            }))
            
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.fusion_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
    def forward(self, x, speakers=None):
        B, S, _ = x.shape
        if self.adim > 0: h_a = self.proj_a(x[:, :, :self.adim].transpose(1, 2)).transpose(1, 2)
        else: h_a = None 
        if self.tdim > 0: h_t = self.proj_t(x[:, :, self.adim:self.adim+self.tdim].transpose(1, 2)).transpose(1, 2)
        else: h_t = None
        if self.vdim > 0: h_v = self.proj_v(x[:, :, self.adim+self.tdim:].transpose(1, 2)).transpose(1, 2)
        else: h_v = None
            
        ref = h_a if h_a is not None else (h_t if h_t is not None else h_v)
        if h_a is None: h_a = torch.zeros_like(ref)
        if h_t is None: h_t = torch.zeros_like(ref)
        if h_v is None: h_v = torch.zeros_like(ref)
        
        if speakers is not None:
            spk_h = self.speaker_emb(speakers)
            h_a = h_a + spk_h; h_t = h_t + spk_h; h_v = h_v + spk_h
        
        V_stack = torch.stack([h_a, h_t, h_v], dim=2) + self.modality_emb
        V = V_stack.reshape(B, 3*S, self.hidden_dim)
        E = self.edge_init(V) 
        
        for layer in self.layers:
            A = layer['adj'](V, E) 
            # [TRICK] Apply DropPath to Residuals
            E_res = layer['agg_attn'](query=E, key=V, value=V, adjacency_mask=A)
            E = layer['norm1'](E + layer['drop_path'](E_res))
            E = E + layer['drop_path'](layer['ffn_e'](E))
            
            V_res = layer['dist_attn'](query=V, key=E, value=E, adjacency_mask=A.transpose(-2, -1))
            V = layer['norm2'](V + layer['drop_path'](V_res))
            V = V + layer['drop_path'](layer['ffn_v'](V))
            
        V = self.final_norm(V)
        V_seq = V.view(B, S, 3, self.hidden_dim)
        V_concat = torch.cat([V_seq[:,:,0,:], V_seq[:,:,1,:], V_seq[:,:,2,:]], dim=-1)
        out = self.fusion_proj(V_concat) 
        return out, None, None

class ExpertPoolingClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, noise_level=1e-2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.noise_level = noise_level
        self.router_norm = nn.LayerNorm(hidden_dim)
        self.proj_in = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.score_projector = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hyperedges):
        xd = self.router_norm(hyperedges.detach()) 
        xd = self.proj_in(xd)
        router_logits = self.score_projector(xd)
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(router_logits) * self.noise_level
            router_logits = router_logits + noise
        router_probs = torch.sigmoid(router_logits)
        weighted_feat = hyperedges * router_probs
        logits = self.classifier(weighted_feat) 
        return logits, None