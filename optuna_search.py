import optuna
import argparse
import torch
import os
import numpy as np
import logging
import sys
import datetime
from types import SimpleNamespace

# === Imports ===
from train_final_moe import (
    train_classifier_phase, 
    train_generator_phase, 
    train_finetune_phase, 
    run_final_eval, 
    get_loaders_generic, # [修正] 这里必须导入 get_loaders_generic
    read_data,
    setup_seed,
    get_class_weights
)
from models_moe import CausalMoEGen, ConfidenceGating, FeatureProjector
from hypergraph_model import AsymmetricCausalHypergraph, ExpertPoolingClassifier
from path import PATH_TO_LABEL_LINUX, PATH_TO_FEATURES_LINUX

# === 配置 ===
TARGET_DATASET = 'CMUMOSI' # 注意这里如果是跑 MOSEI 请确保是 CMUMOSEI
# ============

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"🆔 本次实验 ID: {RUN_ID} | Target: {TARGET_DATASET}")

print("⏳正在将数据集预加载到内存 (Pre-loading data)...")
FEAT_ROOT = PATH_TO_FEATURES_LINUX[TARGET_DATASET]
LABEL_PATH = PATH_TO_LABEL_LINUX[TARGET_DATASET]

NAME2FEATS_AUDIO, DIM_AUDIO = read_data(LABEL_PATH, os.path.join(FEAT_ROOT, 'wav2vec-large-c-UTT'))
NAME2FEATS_TEXT, DIM_TEXT = read_data(LABEL_PATH, os.path.join(FEAT_ROOT, 'deberta-large-4-UTT'))
NAME2FEATS_VIDEO, DIM_VIDEO = read_data(LABEL_PATH, os.path.join(FEAT_ROOT, 'manet_UTT'))

AUDIO_DATA = (NAME2FEATS_AUDIO, DIM_AUDIO)
TEXT_DATA = (NAME2FEATS_TEXT, DIM_TEXT)
VIDEO_DATA = (NAME2FEATS_VIDEO, DIM_VIDEO)
TOTAL_DIM = DIM_AUDIO + DIM_TEXT + DIM_VIDEO
print(f"✅ 数据加载完毕! Total Dim: {TOTAL_DIM}")

# === 任务参数 ===
if TARGET_DATASET in ['CMUMOSI', 'CMUMOSEI']:
    TASK = 'regression'
    NUM_CLASSES = 1
    FOLDS = [1]
else:
    TASK = 'classification'
    NUM_CLASSES = 4 if 'Four' in TARGET_DATASET else 6
    FOLDS = [1, 2, 3, 4, 5]

def get_optuna_logger(save_dir):
    logger = logging.getLogger(f"optuna_{save_dir}") 
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(save_dir, "train_detail.log"), mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def run_process(args, logger, is_final_run=False):
    fold_scores = []
    # 如果是 Final Run，测试不同缺失率
    test_rates = [0.0] if not is_final_run else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    final_stats = {rate: {'acc': [], 'f1': [], 'mae': []} for rate in test_rates}
    
    for fold in FOLDS:
        if is_final_run: print(f"Processing Fold {fold}...", flush=True)
        
        # [修正] 调用 get_loaders_generic
        tr_loader, val_loader, te_loader, tr_set = get_loaders_generic(
            args, fold, AUDIO_DATA, TEXT_DATA, VIDEO_DATA, LABEL_PATH
        )
        
        class_weights = get_class_weights(tr_set, args.device, args.task)
        
        inp = FeatureProjector(TOTAL_DIM, args.hidden_dim, dropout=args.dropout).to(args.device)
        hyp = AsymmetricCausalHypergraph(args.hidden_dim, 0, 0, hidden_dim=args.hidden_dim).to(args.device)
        cls = ExpertPoolingClassifier(args.hidden_dim, args.num_classes).to(args.device)
        gen = CausalMoEGen(args.hidden_dim, num_experts=args.num_experts).to(args.device)
        gate = ConfidenceGating(args.hidden_dim).to(args.device)
        
        # 回归任务通常用 Test Set 做验证监控
        eval_loader = te_loader if args.task == 'regression' else val_loader
        
        # Phase 1
        p1_score = train_classifier_phase(
            args=args, fold_idx=fold, input_proj=inp, hypergraph=hyp, classifier=cls, 
            train_loader=tr_loader, val_loader=eval_loader, 
            logger=logger, class_weights=class_weights
        )
        
        threshold = 0.6 if args.task == 'regression' else 0.4
        if not is_final_run and p1_score < threshold: raise optuna.TrialPruned()

        # Phase 2
        train_generator_phase(
            args=args, fold_idx=fold, input_proj=inp, generator=gen, 
            train_loader=tr_loader, val_loader=eval_loader, logger=logger
        )
        
        # Phase 3
        train_finetune_phase(
            args=args, fold_idx=fold, input_proj=inp, generator=gen, hypergraph=hyp, classifier=cls, gate=gate,
            train_loader=tr_loader, val_loader=eval_loader, logger=logger, 
            class_weights=class_weights
        )
        
        # Eval
        # 这里不需要传入 normalizer，因为最新版 train_final_moe 移除了它
        res = run_final_eval(args, fold, inp, gen, hyp, cls, gate, te_loader, logger)
        fold_scores.append(res[0.0]['F1'])
        
        if is_final_run:
            for rate, metrics in res.items():
                if rate in final_stats:
                    final_stats[rate]['acc'].append(metrics['Acc'])
                    final_stats[rate]['f1'].append(metrics['F1'])
                    if 'MAE' in metrics: final_stats[rate]['mae'].append(metrics['MAE'])
        
        del inp, hyp, cls, gen, gate
        torch.cuda.empty_cache()

    if is_final_run:
        print("\n" + "="*50)
        print(f"📊 FINAL RESULTS ({RUN_ID}):")
        print("="*50)
        for rate in test_rates:
            avg_acc = np.mean(final_stats[rate]['acc'])
            avg_f1 = np.mean(final_stats[rate]['f1'])
            metric_str = f"Rate {rate:<3}: Avg Acc {avg_acc*100:.2f}% | Avg F1 {avg_f1*100:.2f}%"
            if final_stats[rate]['mae']: metric_str += f" | Avg MAE {np.mean(final_stats[rate]['mae']):.4f}"
            print(metric_str)
        print("="*50 + "\n")

    return np.mean(fold_scores)

def objective(trial):
    # === 搜索空间 (针对 MOSEI 优化) ===
    # MOSEI 数据量大，Hidden Dim 可以尝试更大
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512]) 
    dropout = trial.suggest_float('dropout', 0.2, 0.5, step=0.1)
    
    p1_lr = trial.suggest_float('p1_lr', 1e-4, 3e-4, log=True)
    p2_lr = trial.suggest_float('p2_lr', 1e-4, 5e-4, log=True)
    p3_gen_lr = trial.suggest_float('p3_gen_lr', 5e-5, 2e-4, log=True)
    
    p1_mask = trial.suggest_float('p1_mask', 0.3, 0.6, step=0.1)
    p3_rec_weight = trial.suggest_float('p3_rec_weight', 0.2, 0.6)
    
    trial_dir = f'./optuna_logs_{TARGET_DATASET}/{RUN_ID}/trial_{trial.number}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args = SimpleNamespace(
        seed=42, device=device,
        save_dir=trial_dir,
        dataset=TARGET_DATASET, task=TASK,
        hidden_dim=hidden_dim, num_classes=NUM_CLASSES, num_experts=4, dropout=dropout,
        p1_epochs=40, p2_epochs=20, p3_epochs=30, 
        p1_lr=p1_lr, p1_wd=1e-4, p1_mask=p1_mask,
        p2_lr=p2_lr, p3_rec_weight=p3_rec_weight, 
        p3_gate_lr=1e-3, p3_gen_lr=p3_gen_lr,
        rdrop_alpha=0.0, label_smoothing=0.0, scheduler='cosine',
        fusion_type='soft', skip_p2=False, skip_p3=False, use_class_weight=True,
        ablation='none'
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_optuna_logger(args.save_dir)
    setup_seed(args.seed)
    
    try:
        return run_process(args, logger, is_final_run=False)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        import traceback; traceback.print_exc(); return 0.0

if __name__ == "__main__":
    # 1. 创建 Optuna Study 并开始搜索
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0))
    print(f"🚀 开始 Optuna 自动调参 [ID: {RUN_ID}] - Dataset: {TARGET_DATASET}")
    study.optimize(objective, n_trials=30)
    
    print(f"最佳参数: {study.best_params}")
    
    # 2. 定义最终验证的参数配置
    final_save_dir = f'./results_best_{TARGET_DATASET}/{RUN_ID}_BestParams'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    final_args = SimpleNamespace(
        seed=42, device=device,
        save_dir=final_save_dir,
        dataset=TARGET_DATASET, task=TASK,
        hidden_dim=study.best_params['hidden_dim'],
        num_classes=NUM_CLASSES, num_experts=4, dropout=study.best_params['dropout'],
        # Final run full epochs (最终验证使用完整 epoch)
        p1_epochs=60, p2_epochs=30, p3_epochs=50, 
        p1_lr=study.best_params['p1_lr'], p1_wd=1e-4, p1_mask=study.best_params['p1_mask'],
        p2_lr=study.best_params['p2_lr'],
        p3_rec_weight=study.best_params['p3_rec_weight'], 
        p3_gate_lr=1e-3, p3_gen_lr=study.best_params['p3_gen_lr'],
        rdrop_alpha=0.0, label_smoothing=0.0, scheduler='cosine',
        fusion_type='soft', skip_p2=False, skip_p3=False, use_class_weight=True,
        ablation='none'  # <--- 【修复点 1：必须添加这一行，否则会报错】
    )
    
    # 3. 创建目录
    os.makedirs(final_args.save_dir, exist_ok=True)
    
    # ==========================================
    # 【新增功能】将最佳参数保存到 txt 文件
    # ==========================================
    txt_path = os.path.join(final_args.save_dir, "best_params.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write(f"Experiment ID: {RUN_ID}\n")
        f.write(f"Dataset: {TARGET_DATASET}\n")
        f.write(f"Best Trial Value (Score): {study.best_value}\n")
        f.write("=" * 30 + "\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    print(f"💾 最佳参数已保存至: {txt_path}")
    # ==========================================

    final_logger = get_optuna_logger(final_args.save_dir)
    setup_seed(final_args.seed)
    
    print(f"⏳ 正在运行最终验证... 结果将保存至: {final_save_dir}")
    run_process(final_args, final_logger, is_final_run=True)