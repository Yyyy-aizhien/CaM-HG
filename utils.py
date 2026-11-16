import torch
import random
import numpy as np
import logging
import os
import sys
from sklearn.metrics import accuracy_score, f1_score

def setup_seed(seed):
    """设置全局随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(save_dir, filename="train.log"):
    """初始化日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    
    # 清除已有的 Handler，防止重复打印
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 格式：时间 - 消息
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 1. 文件输出
    fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 2. 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

class Metrics:
    """标准评估指标"""
    def __init__(self):
        pass

    def eval_iemocap(self, y_pred, y_true):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred_cls = np.argmax(y_pred, axis=1)
        
        acc = accuracy_score(y_true, y_pred_cls)
        f1 = f1_score(y_true, y_pred_cls, average='weighted')
        
        return {"Acc": round(acc, 4), "F1": round(f1, 4)}

    def save_checkpoint(self, state, filename="checkpoint.pth"):
        torch.save(state, filename)

class EarlyStopping:
    """早停策略：当验证集指标在 patience 轮内没有提升时停止训练"""
    def __init__(self, patience=10, verbose=False, delta=0, mode='max'):
        """
        Args:
            patience (int): 上次指标提升后，等待多少个 epoch 仍无提升则停止。
            verbose (bool): 是否打印日志。
            delta (float): 指标变化的最小阈值，小于此值的变化不被认为是提升。
            mode (str): 'max' 表示指标越大越好 (如 Acc)，'min' 表示越小越好 (如 Loss)。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.val_score_min = np.Inf

    def __call__(self, score, logger=None):
        if self.best_score is None:
            self.best_score = score
        elif self.is_worse(score):
            self.counter += 1
            if logger:
                logger.info(f'⚠️  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    
    def is_worse(self, score):
        if self.mode == 'max':
            return score <= self.best_score + self.delta
        else:
            return score >= self.best_score - self.delta