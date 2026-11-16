import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter
# [修改 1] 引入 confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# === Imports ===
from dataloader_iemocap import IEMOCAPDataset, collate_fn, read_data
from models_moe import CausalMoEGen, ConfidenceGating, FeatureProjector
from hypergraph_model import AsymmetricCausalHypergraph, ExpertPoolingClassifier
from utils import Metrics, setup_seed, get_logger, EarlyStopping
from path import PATH_TO_LABEL_LINUX, PATH_TO_FEATURES_LINUX

# === [NEW] Baseline Model for Ablation (w/o Hypergraph) ===
class TransformerBaseline(nn.Module):
    """
    用于消融实验的 Baseline：
    移除了超图和动态边结构，仅使用标准的 Transformer Encoder 进行融合。
    """
    def __init__(self, hidden_dim, layers=2):
        super().__init__()
        # 使用标准 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.speaker_emb = nn.Embedding(4, hidden_dim)
        # 为了保持接口一致，需要一个输出投影
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, speakers=None):
        # x: [B, S, D]
        if speakers is not None:
            x = x + self.speaker_emb(speakers)
        
        # 标准 Self-Attention (没有 Block Causal Mask 的精细控制)
        out = self.transformer(x)
        out = self.fusion_proj(out)
        
        # 返回格式需与 Hypergraph 保持一致: (out, None, None)
        return out, None, None

# ==========================================================

def get_args():
    parser = argparse.ArgumentParser(description='MCMoE SOTA (MOSEI Fix)')
    
    # Base
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./results_mosi_sota_final')
    parser.add_argument('--folds', type=str, default='1,2,3,4,5')
    
    # Data
    parser.add_argument('--dataset', type=str, default='CMUMOSI', 
                        choices=['IEMOCAPSix', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI'])
    parser.add_argument('--audio_feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text_feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video_feature', type=str, default='manet_UTT')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.45) 
    parser.add_argument('--num_classes', type=int, default=0) 
    parser.add_argument('--num_experts', type=int, default=4)
    
    # Training Params
    parser.add_argument('--p1_epochs', type=int, default=50)
    parser.add_argument('--p1_lr', type=float, default=0.000177)
    parser.add_argument('--p1_wd', type=float, default=0.001)
    parser.add_argument('--p1_mask', type=float, default=0.4)
    
    parser.add_argument('--p2_epochs', type=int, default=30)
    parser.add_argument('--p2_lr', type=float, default=0.000173)
    
    parser.add_argument('--p3_epochs', type=int, default=40)
    parser.add_argument('--p3_gen_lr', type=float, default=5.777116873457687e-05)
    parser.add_argument('--p3_gate_lr', type=float, default=0.0031)
    parser.add_argument('--p3_rec_weight', type=float, default=0.10656786065970517)
    
    # Tricks
    parser.add_argument('--rdrop_alpha', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    parser.add_argument('--fusion_type', type=str, default='soft')
    parser.add_argument('--use_class_weight', action='store_true', default=True)
    parser.add_argument('--skip_p2', action='store_true')
    parser.add_argument('--skip_p3', action='store_true')

    # === [NEW] Ablation Study Arguments ===
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_imputation', 'no_moe', 'no_gating', 'no_hypergraph', 'no_causal_factor'],
                        help='Choose ablation mode: \n'
                             'none: Full Model\n'
                             'no_imputation: Remove Generator\n'
                             'no_moe: Use Single Expert\n'
                             'no_gating: Remove Confidence Gating\n'
                             'no_hypergraph: Replace Hypergraph with Standard Transformer\n'
                             'no_causal_factor: Remove Causal Factor Extractor in Gen')

    try: args = parser.parse_args()
    except: args = parser.parse_args([])
    
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_classes = 1; args.task = 'regression'
        print(f"👉 Detected {args.dataset}: Setting task to Regression (num_classes=1)")
    else:
        args.task = 'classification'
        if args.dataset == 'IEMOCAPSix': args.num_classes = 6
        else: args.num_classes = 4
        
    return args

def get_class_weights(dataset, device, task):
    if task == 'regression': return None
    all_labels = []
    for item in dataset.data_list: all_labels.extend(item['labels'].tolist())
    valid_labels = [l for l in all_labels if l >= 0]
    if not valid_labels: return torch.ones(1).to(device)
    counts = Counter(valid_labels)
    num_classes = max(valid_labels) + 1
    total = len(valid_labels)
    weights = torch.zeros(num_classes).to(device)
    for cls_idx in range(num_classes):
        if cls_idx in counts: weights[cls_idx] = total / (len(counts) * counts[cls_idx])
        else: weights[cls_idx] = 1.0 
    return weights

def masked_mse_loss(preds, labels, mask):
    loss = F.mse_loss(preds, labels, reduction='none')
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)

def eval_regression(y_pred, y_true, mask=None):
    y_pred = y_pred.cpu().numpy().reshape(-1)
    y_true = y_true.cpu().numpy().reshape(-1)
    if mask is not None:
        mask = mask.cpu().numpy().reshape(-1)
        y_pred = y_pred[mask == 1]
        y_true = y_true[mask == 1]
    
    non_zeros = (y_true != 0)
    y_true_nz = y_true[non_zeros]
    y_pred_nz = y_pred[non_zeros]
    
    if len(y_true_nz) == 0: return {"Acc": 0.0, "F1": 0.0, "MAE": 0.0}

    y_true_bin = (y_true_nz > 0).astype(int)
    y_pred_bin = (y_pred_nz > 0).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {"Acc": round(acc, 4), "F1": round(f1, 4), "MAE": round(mae, 4)}

def apply_mask_in_loop(audio, text, video, missing_rate, device):
    B, S, _ = audio.shape
    
    if missing_rate == 0.0:
        mask = torch.ones(B, S, 3).to(device)
    else:
        mask = (torch.rand(B, S, 3) > missing_rate).float().to(device)
    
    # Targeted Blindness Strategy
    if missing_rate > 0:
        sample_rand = torch.rand(B, device=device)
        drop_indices = sample_rand < 0.25 
        mask[drop_indices, :, 1] = 0.0

    all_zeros = (mask.sum(dim=-1) == 0)
    if all_zeros.any():
        rand_idx = torch.randint(0, 3, (all_zeros.sum(),), device=device)
        b_idx, s_idx = torch.nonzero(all_zeros, as_tuple=True)
        mask[b_idx, s_idx, rand_idx] = 1.0
        
    m_audio = audio * mask[:, :, 0].unsqueeze(-1)
    m_text = text * mask[:, :, 1].unsqueeze(-1)
    m_video = video * mask[:, :, 2].unsqueeze(-1)
    
    return m_audio, m_text, m_video, mask

def apply_test_mask_fixed(audio, text, video, missing_rate, device):
    B, S, _ = audio.shape
    if missing_rate == 0.0:
        return audio, text, video, torch.ones(B, S, 3).to(device)
    mask = (torch.rand(B, S, 3) > missing_rate).float().to(device)
    all_zeros = (mask.sum(dim=-1) == 0)
    if all_zeros.any():
        indices = torch.randint(0, 3, (all_zeros.sum(),), device=device)
        mask[all_zeros, indices] = 1.0
    m_audio = audio * mask[:, :, 0].unsqueeze(-1)
    m_text = text * mask[:, :, 1].unsqueeze(-1)
    m_video = video * mask[:, :, 2].unsqueeze(-1)
    return m_audio, m_text, m_video, mask

def get_loaders_generic(args, fold_idx, audio, text, video, label_path):
    is_regression = (args.task == 'regression')
    if is_regression:
        try:
            data = pickle.load(open(label_path, "rb"), encoding='latin1')
            if isinstance(data, dict):
                train_vids = list(data.get('trainVid', []))
                valid_vids = list(data.get('validVid', []))
                test_vids = list(data.get('testVid', []))
            else:
                if len(data) >= 7:
                    train_vids, valid_vids, test_vids = list(data[4]), list(data[5]), list(data[6])
                else:
                    train_vids, test_vids, valid_vids = list(data[4]), list(data[5]), [] 
            if len(valid_vids) == 0:
                print("⚠️ No validation split found. Using last 10% of train.")
                split = int(len(train_vids) * 0.9)
                valid_vids = train_vids[split:]
                train_vids = train_vids[:split]
        except Exception as e: print(f"Error: {e}"); raise
    else:
        videoIDs, _, _, _, _, _ = pickle.load(open(label_path, "rb"), encoding='latin1')
        all_vids = sorted(list(videoIDs.keys()))
        train_vids, test_vids = [], []
        target_ses = f"Ses0{fold_idx}"
        for vid in all_vids:
            if target_ses in vid: test_vids.append(vid)
            else: train_vids.append(vid)
        valid_vids = test_vids 

    tr_set = IEMOCAPDataset(label_path, audio, text, video, mask_ratio=0.0, specific_vids=train_vids, task_type=args.task)
    val_set = IEMOCAPDataset(label_path, audio, text, video, mask_ratio=0.0, specific_vids=valid_vids, task_type=args.task)
    te_set = IEMOCAPDataset(label_path, audio, text, video, mask_ratio=0.0, specific_vids=test_vids, task_type=args.task)

    def count_utt(ds, name):
        count = sum([len(item['labels']) for item in ds.data_list])
        print(f"📊 {name} Set: {len(ds)} Videos | {count} Utterances")
    
    print("-" * 40)
    count_utt(tr_set, "Train"); count_utt(val_set, "Valid"); count_utt(te_set, "Test ")
    print("-" * 40)

    tr_loader = DataLoader(tr_set, 32, True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, 32, False, collate_fn=collate_fn)
    te_loader = DataLoader(te_set, 32, False, collate_fn=collate_fn)
    return tr_loader, val_loader, te_loader, tr_set

def read_feats(args):
    feat_root = PATH_TO_FEATURES_LINUX[args.dataset]
    label_path = PATH_TO_LABEL_LINUX[args.dataset]
    print(f"📂 Loading features from: {feat_root}")
    a = read_data(label_path, os.path.join(feat_root, args.audio_feature))
    t = read_data(label_path, os.path.join(feat_root, args.text_feature))
    v = read_data(label_path, os.path.join(feat_root, args.video_feature))
    return (a[0], a[1]), (t[0], t[1]), (v[0], v[1]), label_path

# --- Training Phases ---

def train_classifier_phase(args, fold_idx, input_proj, hypergraph, classifier, train_loader, val_loader, logger, class_weights=None):
    logger.info(f"\n>>> 🚀 PHASE 1 (Fold {fold_idx}): Main Training (Task={args.task}) <<<")
    optimizer = optim.AdamW(list(input_proj.parameters()) + list(hypergraph.parameters()) + list(classifier.parameters()), lr=args.p1_lr, weight_decay=args.p1_wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.p1_epochs, eta_min=1e-6)
    
    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, label_smoothing=args.label_smoothing)
    else: criterion = None
    
    best_score = 0.0
    for epoch in range(args.p1_epochs):
        input_proj.train(); hypergraph.train(); classifier.train()
        avg_loss = 0; steps = 0
        pbar = tqdm(train_loader, desc=f"F{fold_idx} P1 Ep {epoch+1}", leave=False)
        for batch in pbar:
            gt_a = batch['audio_gt'].to(args.device); gt_t = batch['text_gt'].to(args.device); gt_v = batch['visual_gt'].to(args.device)
            labels, speakers = batch['labels'].to(args.device), batch['speakers'].to(args.device)
            if gt_a.size(1) > 60: gt_a, gt_t, gt_v = gt_a[:,:60], gt_t[:,:60], gt_v[:,:60]; labels, speakers = labels[:,:60], speakers[:,:60]

            m_a, m_t, m_v, _ = apply_mask_in_loop(gt_a, gt_t, gt_v, args.p1_mask, args.device)
            
            optimizer.zero_grad()
            latent = input_proj(torch.cat([m_a, m_t, m_v], dim=-1))
            logits, _ = classifier(hypergraph(latent, speakers)[0])
            
            if args.task == 'regression':
                loss = masked_mse_loss(logits.squeeze(-1), labels, (labels != -100.0).float())
            else:
                loss = criterion(logits.reshape(-1, args.num_classes), labels.reshape(-1))
            
            loss.backward(); optimizer.step()
            avg_loss += loss.item(); steps += 1
            pbar.set_postfix({'L': f"{loss.item():.2f}"})
        
        # Validation
        input_proj.eval(); hypergraph.eval(); classifier.eval()
        all_preds, all_labels, all_masks = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                gt_a = batch['audio_gt'].to(args.device); gt_t = batch['text_gt'].to(args.device); gt_v = batch['visual_gt'].to(args.device)
                labels, speakers = batch['labels'].to(args.device), batch['speakers'].to(args.device)
                if gt_a.size(1) > 60: gt_a, gt_t, gt_v = gt_a[:,:60], gt_t[:,:60], gt_v[:,:60]; labels, speakers = labels[:,:60], speakers[:,:60]
                latent = input_proj(torch.cat([gt_a, gt_t, gt_v], dim=-1))
                logits, _ = classifier(hypergraph(latent, speakers)[0])
                
                if args.task == 'regression':
                    # [修复] Flatten tensors before appending to list
                    # 避免因不同 batch sequence length 导致的 cat error
                    all_preds.append(logits.squeeze(-1).reshape(-1))
                    all_labels.append(labels.reshape(-1))
                    all_masks.append((labels != -100.0).float().reshape(-1))
                else:
                    all_preds.append(logits[(labels != -1)])
                    all_labels.append(labels[(labels != -1)])

        if args.task == 'regression':
            score = eval_regression(torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_masks))['F1']
            display = f"F1: {score:.4f}"
        else:
            score = Metrics().eval_iemocap(torch.cat(all_preds), torch.cat(all_labels))['F1']
            display = f"F1: {score:.4f}"

        logger.info(f"F{fold_idx} P1 Ep {epoch+1:02d} | Val {display}")
        if score > best_score:
            best_score = score
            torch.save({'inp': input_proj.state_dict(), 'hyp': hypergraph.state_dict(), 'cls': classifier.state_dict()}, os.path.join(args.save_dir, f'fold{fold_idx}_phase1.pth'))
        scheduler.step()
    return best_score

def train_generator_phase(args, fold_idx, input_proj, generator, train_loader, val_loader, logger):
    if args.skip_p2: return
    logger.info(f"\n>>> 🚀 PHASE 2 (Fold {fold_idx}): Generator Pre-training <<<")
    ckpt = torch.load(os.path.join(args.save_dir, f'fold{fold_idx}_phase1.pth'))
    input_proj.load_state_dict(ckpt['inp']); input_proj.eval()
    for p in input_proj.parameters(): p.requires_grad = False
    
    optimizer = optim.AdamW(generator.parameters(), lr=args.p2_lr, weight_decay=1e-4)
    best_loss = float('inf')
    
    for epoch in range(args.p2_epochs):
        generator.train()
        avg_loss = 0; steps = 0
        pbar = tqdm(train_loader, desc=f"F{fold_idx} P2 Ep {epoch+1}", leave=False)
        for batch in pbar:
            gt_a = batch['audio_gt'].to(args.device); gt_t = batch['text_gt'].to(args.device); gt_v = batch['visual_gt'].to(args.device)
            labels = batch['labels'].to(args.device)
            if gt_a.size(1) > 60: gt_a, gt_t, gt_v = gt_a[:,:60], gt_t[:,:60], gt_v[:,:60]; labels = labels[:,:60]
            
            m_a, m_t, m_v, _ = apply_mask_in_loop(gt_a, gt_t, gt_v, 0.6, args.device)
            with torch.no_grad():
                gt_latent = input_proj(torch.cat([gt_a, gt_t, gt_v], dim=-1))
                m_latent = input_proj(torch.cat([m_a, m_t, m_v], dim=-1))
            
            pred_latent = generator(x_curr=m_latent)
            mask = (labels != -100.0).float().unsqueeze(-1) if args.task == 'regression' else (labels != -1).float().unsqueeze(-1)
            loss = (F.mse_loss(pred_latent, gt_latent, reduction='none')*mask).sum() / (mask.sum() + 1e-8)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            avg_loss += loss.item(); steps += 1
        
        if (epoch+1) % 10 == 0: logger.info(f"P2 Ep {epoch+1} Loss: {avg_loss/steps:.4f}")
        if avg_loss/steps < best_loss:
            best_loss = avg_loss/steps
            torch.save(generator.state_dict(), os.path.join(args.save_dir, f'fold{fold_idx}_phase2.pth'))

def train_finetune_phase(args, fold_idx, input_proj, generator, hypergraph, classifier, gate, train_loader, val_loader, logger, class_weights=None):
    if args.skip_p3: return
    logger.info(f"\n>>> 🚀 PHASE 3 (Fold {fold_idx}): Joint Fine-tuning (Chaos Mode) | Ablation: {args.ablation} <<<")
    p1 = torch.load(os.path.join(args.save_dir, f'fold{fold_idx}_phase1.pth'))
    input_proj.load_state_dict(p1['inp']); hypergraph.load_state_dict(p1['hyp']); classifier.load_state_dict(p1['cls'])
    if not args.skip_p2:
        p2 = torch.load(os.path.join(args.save_dir, f'fold{fold_idx}_phase2.pth'))
        generator.load_state_dict(p2)
    
    for p in input_proj.parameters(): p.requires_grad = False
    input_proj.eval()
    for p in generator.parameters(): p.requires_grad = True
    
    optimizer = optim.AdamW([
        {'params': hypergraph.parameters(), 'lr': 1e-5},
        {'params': classifier.parameters(), 'lr': 1e-5},
        {'params': generator.parameters(), 'lr': args.p3_gen_lr},
        {'params': gate.parameters(), 'lr': args.p3_gate_lr}
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.p3_epochs, eta_min=1e-7)
    
    best_score = 0.0
    if args.task == 'regression': criterion = nn.MSELoss(reduction='none')
    else: criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    for epoch in range(args.p3_epochs):
        hypergraph.train(); classifier.train(); generator.train(); gate.train()
        acc_gate = 0; steps = 0
        pbar = tqdm(train_loader, desc=f"F{fold_idx} P3 Ep {epoch+1}", leave=False)
        for batch in pbar:
            gt_a = batch['audio_gt'].to(args.device); gt_t = batch['text_gt'].to(args.device); gt_v = batch['visual_gt'].to(args.device)
            labels, speakers = batch['labels'].to(args.device), batch['speakers'].to(args.device)
            if gt_a.size(1) > 60: gt_a, gt_t, gt_v = gt_a[:,:60], gt_t[:,:60], gt_v[:,:60]; labels, speakers = labels[:,:60], speakers[:,:60]
            
            curr_mask_rate = np.random.uniform(0.0, 1.0)
            m_a, m_t, m_v, _ = apply_mask_in_loop(gt_a, gt_t, gt_v, curr_mask_rate, args.device)
            
            with torch.no_grad():
                m_z = input_proj(torch.cat([m_a, m_t, m_v], dim=-1))
                gt_z = input_proj(torch.cat([gt_a, gt_t, gt_v], dim=-1))
            
            # === [NEW] Ablation Logic ===
            gen_z = generator(x_curr=m_z)
            
            if args.ablation == 'no_imputation':
                final_z = m_z 
                loss_rec = torch.tensor(0.0).to(args.device)
            elif args.ablation == 'no_gating':
                final_z = gen_z 
                if args.task == 'regression': mask_rec = (labels != -100.0).float().unsqueeze(-1)
                else: mask_rec = (labels != -1).float().unsqueeze(-1)
                loss_rec = (F.mse_loss(gen_z, gt_z, reduction='none')*mask_rec).sum()/mask_rec.sum()
            else:
                # 标准逻辑 (CaM-HG Full)
                _, g_val = gate(gen_z)
                final_z = (1-g_val)*m_z + g_val*gen_z
                if args.task == 'regression': mask_rec = (labels != -100.0).float().unsqueeze(-1)
                else: mask_rec = (labels != -1).float().unsqueeze(-1)
                loss_rec = (F.mse_loss(gen_z, gt_z, reduction='none')*mask_rec).sum()/mask_rec.sum()
            # ==============================
            
            logits, _ = classifier(hypergraph(final_z, speakers)[0])
            
            if args.task == 'regression':
                mask = (labels != -100.0).float()
                loss_cls = masked_mse_loss(logits.squeeze(-1), labels, mask)
            else:
                loss_cls = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)(logits.reshape(-1, args.num_classes), labels.reshape(-1))
            
            dynamic_weight = args.p3_rec_weight * (1.0 + curr_mask_rate)
            loss = loss_cls + dynamic_weight * loss_rec
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix({'L': f"{loss.item():.2f}"})
            
        # Eval
        hypergraph.eval(); classifier.eval(); generator.eval(); gate.eval()
        all_preds, all_labels, all_masks = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                gt_a = batch['audio_gt'].to(args.device); gt_t = batch['text_gt'].to(args.device); gt_v = batch['visual_gt'].to(args.device)
                labels, speakers = batch['labels'].to(args.device), batch['speakers'].to(args.device)
                if gt_a.size(1) > 60: gt_a, gt_t, gt_v = gt_a[:,:60], gt_t[:,:60], gt_v[:,:60]; labels, speakers = labels[:,:60], speakers[:,:60]
                latent = input_proj(torch.cat([gt_a, gt_t, gt_v], dim=-1))
                logits, _ = classifier(hypergraph(latent, speakers)[0])
                
                if args.task == 'regression':
                    all_preds.append(logits.squeeze(-1).reshape(-1))
                    all_labels.append(labels.reshape(-1))
                    all_masks.append((labels != -100.0).float().reshape(-1))
                else:
                    all_preds.append(logits[labels!=-1]); all_labels.append(labels[labels!=-1])
        
        if args.task == 'regression':
            score = eval_regression(torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_masks))['F1']
        else:
            score = Metrics().eval_iemocap(torch.cat(all_preds), torch.cat(all_labels))['F1']
            
        logger.info(f"P3 Ep {epoch+1} Val F1: {score:.4f}")
        if score > best_score:
            best_score = score
            torch.save({'hyp': hypergraph.state_dict(), 'cls': classifier.state_dict(), 'gen': generator.state_dict(), 'gate': gate.state_dict()}, os.path.join(args.save_dir, f'fold{fold_idx}_phase3.pth'))
        scheduler.step()
    logger.info(f"✅ F{fold_idx} P3 Best F1: {best_score:.4f}")

def run_final_eval(args, fold_idx, input_proj, generator, hypergraph, classifier, gate, test_loader, logger):
    logger.info(f"\n🛡️ FINAL EVAL (Fold {fold_idx}) | Ablation: {args.ablation}")
    p3 = torch.load(os.path.join(args.save_dir, f'fold{fold_idx}_phase3.pth'))
    input_proj.load_state_dict(torch.load(os.path.join(args.save_dir, f'fold{fold_idx}_phase1.pth'))['inp'])
    hypergraph.load_state_dict(p3['hyp']); classifier.load_state_dict(p3['cls'])
    generator.load_state_dict(p3['gen']); gate.load_state_dict(p3['gate'])
    
    input_proj.eval(); generator.eval(); hypergraph.eval(); classifier.eval(); gate.eval()
    
    results = {}
    test_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for rate in test_rates:
        all_preds, all_labels, all_masks = [], [], []
        for batch in test_loader:
            with torch.no_grad():
                gt_a = batch['audio_gt'].to(args.device); gt_t = batch['text_gt'].to(args.device); gt_v = batch['visual_gt'].to(args.device)
                labels, speakers = batch['labels'].to(args.device), batch['speakers'].to(args.device)
                if gt_a.size(1) > 60: gt_a, gt_t, gt_v = gt_a[:,:60], gt_t[:,:60], gt_v[:,:60]; labels, speakers = labels[:,:60], speakers[:,:60]
                
                m_a, m_t, m_v, _ = apply_mask_in_loop(gt_a, gt_t, gt_v, rate, args.device)
                m_latent = input_proj(torch.cat([m_a, m_t, m_v], dim=-1))
                
                # === [NEW] Ablation Logic ===
                if args.ablation == 'no_imputation':
                    final_input = m_latent 
                else:
                    raw_gen = generator(x_curr=m_latent)
                    if args.ablation == 'no_gating':
                        final_input = raw_gen
                    else:
                        _, gate_val = gate(raw_gen)
                        final_input = (1 - gate_val) * m_latent + gate_val * raw_gen
                # ==============================
                
                logits, _ = classifier(hypergraph(final_input, speakers)[0])
                
                if args.task == 'regression':
                    all_preds.append(logits.squeeze(-1).reshape(-1))
                    all_labels.append(labels.reshape(-1))
                    all_masks.append((labels != -100.0).float().reshape(-1))
                else:
                    all_preds.append(logits[labels!=-1]); all_labels.append(labels[labels!=-1])
        
        if args.task == 'regression':
            res = eval_regression(torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_masks))
        else:
            res = Metrics().eval_iemocap(torch.cat(all_preds), torch.cat(all_labels))
            
        logger.info(f"Rate {rate:<3}: Acc {res['Acc']*100:.2f}% | F1 {res['F1']*100:.2f}%")
        results[rate] = res
    return results

# === [NEW] Confusion Matrix Extraction Function ===
def save_confusion_matrix_data(args, fold_idx, input_proj, generator, hypergraph, classifier, gate, test_loader, save_dir):
    print(f"\n>>> 正在提取混淆矩阵数据 (Fold {fold_idx}, Dataset: {args.dataset})...")
    
    # Path to Phase 3 checkpoint
    p3_path = os.path.join(save_dir, f'fold{fold_idx}_phase3.pth')
    if not os.path.exists(p3_path):
        print(f"⚠️ 未找到权重文件 {p3_path}，跳过提取。")
        return

    # Load weights
    p3 = torch.load(p3_path)
    p1 = torch.load(os.path.join(save_dir, f'fold{fold_idx}_phase1.pth'))
    
    input_proj.load_state_dict(p1['inp'])
    hypergraph.load_state_dict(p3['hyp'])
    classifier.load_state_dict(p3['cls'])
    generator.load_state_dict(p3['gen'])
    gate.load_state_dict(p3['gate'])
    
    input_proj.eval(); generator.eval(); hypergraph.eval(); classifier.eval(); gate.eval()
    
    # 定义需要保存的缺失率
    target_rates = [0.1, 0.3, 0.5, 0.7]
    cm_results = {}

    for rate in target_rates:
        all_preds = []
        all_labels = []
        
        for batch in test_loader:
            with torch.no_grad():
                gt_a = batch['audio_gt'].to(args.device)
                gt_t = batch['text_gt'].to(args.device)
                gt_v = batch['visual_gt'].to(args.device)
                labels = batch['labels'].to(args.device)
                speakers = batch['speakers'].to(args.device)
                
                if gt_a.size(1) > 60:
                     gt_a, gt_t, gt_v = gt_a[:,:60], gt_t[:,:60], gt_v[:,:60]
                     labels, speakers = labels[:,:60], speakers[:,:60]

                # Apply Mask
                m_a, m_t, m_v, _ = apply_mask_in_loop(gt_a, gt_t, gt_v, rate, args.device)
                
                # Forward
                m_latent = input_proj(torch.cat([m_a, m_t, m_v], dim=-1))
                
                # 默认使用 Full Model 逻辑
                raw_gen = generator(x_curr=m_latent)
                _, gate_val = gate(raw_gen)
                final_input = (1 - gate_val) * m_latent + gate_val * raw_gen
                
                logits, _ = classifier(hypergraph(final_input, speakers)[0])
                
                # Get Preds
                preds = torch.argmax(logits, dim=-1)
                
                # Filter padding
                valid = (labels != -1)
                all_preds.extend(preds[valid].cpu().numpy())
                all_labels.extend(labels[valid].cpu().numpy())
        
        cm_results[rate] = {
            'y_true': np.array(all_labels),
            'y_pred': np.array(all_preds)
        }
        print(f"  - Rate {rate}: Collected {len(all_labels)} samples")

    save_name = f"cm_data_{args.dataset}.npy"
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, cm_results)
    print(f"✅ 数据已保存至 {save_path}")

def main():
    args = get_args()
    setup_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger(args.save_dir)
    logger.info(f"🚀 MCMoE Training (Dataset={args.dataset}, Task={args.task})")
    
    # === [NEW] Handle Ablation Settings ===
    use_causal_flag = True
    
    if args.ablation == 'no_moe':
        args.num_experts = 1
        logger.info("⚠️ Ablation Activated: 'no_moe' -> Setting num_experts = 1")
    
    if args.ablation == 'no_imputation':
        args.skip_p2 = True
        logger.info("⚠️ Ablation Activated: 'no_imputation' -> Skipping Phase 2 & Gen in Phase 3.")
        
    if args.ablation == 'no_causal_factor':
        use_causal_flag = False
        logger.info("⚠️ Ablation Activated: 'no_causal_factor' -> Disabling CausalFactorExtractor in Gen.")

    if args.ablation == 'no_hypergraph':
        logger.info("⚠️ Ablation Activated: 'no_hypergraph' -> Using TransformerBaseline instead of Hypergraph.")
    # ======================================
    
    audio_d, text_d, video_d, label_path = read_feats(args)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']: folds = [1]
    else: folds = [int(f) for f in args.folds.split(',')]

    fold_results = []
    for fold in folds:
        logger.info(f"------ Fold {fold} / Run ------")
        tr_loader, val_loader, te_loader, tr_set = get_loaders_generic(args, fold, audio_d, text_d, video_d, label_path)
        class_weights = get_class_weights(tr_set, args.device, args.task)
        
        dummy_ds = tr_set
        total_dim = dummy_ds.adim + dummy_ds.tdim + dummy_ds.vdim
        
        inp = FeatureProjector(total_dim, args.hidden_dim, dropout=args.dropout).to(args.device)
        
        # === [NEW] Conditional Model Initialization ===
        if args.ablation == 'no_hypergraph':
            hyp = TransformerBaseline(args.hidden_dim, layers=2).to(args.device)
        else:
            hyp = AsymmetricCausalHypergraph(args.hidden_dim, 0, 0, hidden_dim=args.hidden_dim).to(args.device)
        # ==============================================
        
        cls = ExpertPoolingClassifier(args.hidden_dim, args.num_classes).to(args.device)
        
        # === [NEW] Pass Ablation Flag to Gen ===
        gen = CausalMoEGen(args.hidden_dim, num_experts=args.num_experts, use_causal=use_causal_flag).to(args.device)
        # =======================================
        
        gate = ConfidenceGating(args.hidden_dim).to(args.device)
        
        train_classifier_phase(args, fold, inp, hyp, cls, tr_loader, val_loader, logger, class_weights)
        train_generator_phase(args, fold, inp, gen, tr_loader, val_loader, logger)
        train_finetune_phase(args, fold, inp, gen, hyp, cls, gate, tr_loader, val_loader, logger, class_weights)
        
        fold_results.append(run_final_eval(args, fold, inp, gen, hyp, cls, gate, te_loader, logger))
        
        # === [NEW] 提取混淆矩阵数据 (只在第一个或最后一个 Fold 做) ===
        # 只要运行到这里，就会提取该 Fold 的测试集结果
        # 如果你只跑 Fold 1, 这里就会保存 Fold 1 的数据
        save_confusion_matrix_data(args, fold, inp, gen, hyp, cls, gate, te_loader, args.save_dir)
        
    if len(fold_results) > 1:
        logger.info("\n📊 FINAL AVG:")
        for rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            acc = np.mean([f[rate]['Acc'] for f in fold_results])
            f1 = np.mean([f[rate]['F1'] for f in fold_results])
            logger.info(f"Rate {rate:<3}: Acc {acc*100:.2f} | F1 {f1*100:.2f}")

if __name__ == "__main__":
    main()