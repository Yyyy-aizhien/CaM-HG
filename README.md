# CaM-HG

**CaM-HG: Causal-Enhanced MoE and Hypergraphs Network for Incomplete Multimodal Emotion Recognition in Conversations**

**Mingjian Yang, Yong Wang, Peng Liu, Wen Yin\***

*The Laboratory of Intelligent Collaborative Computing, University of Electronic Science and Technology of China*

Source code accompanying the paper. CaM-HG addresses modality missingness in Multimodal Emotion Recognition in Conversation (MERC) with a **restore-then-mine** paradigm: it first *restores* the ruptured fine-grained inter-modal causal chains, then *mines* high-order causal correlations on the restored conversation topology.

## Overview

The framework consists of three main components:

1. **Feature Projector** — projects raw multimodal features (audio / text / visual) into a unified latent space.
2. **Causal-Enhanced MoE Generator** — a Causal Factor Extractor (causal-masked Transformer) conditions N=4 experts (Top-K=2 routing) on historical context to synthesize missing modality features that are causally consistent with the dialogue flow; a **Confidence Gate** then blends observed and restored latents.
3. **Asymmetric Causal Dynamic Hypergraph (ACDH)** — models utterance-modality nodes and hyperedges with a Block Causal Mask (plus a local window), enabling high-order conversational reasoning without future leakage.

Training follows a three-phase curriculum:

| Phase | What it does |
|---|---|
| **P1 — Discriminative Warm-up** | Train projector + hypergraph + classifier with random modality masking (incl. Text Blindness, p=0.25) |
| **P2 — Generative Pre-training** | Freeze projector; pre-train the MoE generator to restore masked latents (self-supervised MSE) |
| **P3 — Joint Fine-tuning** | Missing rate η sampled from U(0,1) per batch; dynamic reconstruction weight λ·(1+η); differential learning rates for generator vs. gate |

## Repository Structure

```
CaM-HG/
├── train_final_moe.py     # Main entry: 3-phase training + evaluation under missing rates 0.0–0.7
├── dataloader_iemocap.py   # Dataset / DataLoaders (IEMOCAP: classification; MOSI/MOSEI: regression)
├── models_moe.py           # FeatureProjector, CausalFactorExtractor, MoEBlock, CausalMoEGen, ConfidenceGating
├── hypergraph_model.py     # AsymmetricCausalHypergraph, ExpertPoolingClassifier
├── hypergraph_layers.py    # CausalDynamicAdjacency, CausalHypergraphAttention, FeedForward, DropPath
├── run_ablations.py        # One-click ablation suite (6 modes)
├── optuna_search.py        # Optional hyperparameter search (Optuna)
├── utils.py                # Seeding, logger, metrics, early stopping
├── config.py               # Auxiliary config holder
├── path.py                 # Dataset path definitions (dataset root = project root /dataset)
└── dataset/                # Place data here (NOT distributed, see below)
```

## Requirements

- Python ≥ 3.8, PyTorch 2.3.0 (experiments ran on a single NVIDIA RTX 4090)
- `pip install -r requirements.txt` (numpy, tqdm, scikit-learn, optuna)

## Dataset Preparation

`path.py` resolves everything relative to the project root, so data must be placed as follows (this is exactly the layout the code expects):

```
CaM-HG/
└── dataset/
    ├── IEMOCAP/                        # 6-class (IEMOCAPSix)
    │   ├── IEMOCAP_features_raw_6way.pkl
    │   └── features/
    │       ├── wav2vec-large-c-UTT/    # audio .npy per utterance (Wav2Vec 2.0 large, 1024-d)
    │       ├── deberta-large-4-UTT/    # text .npy per utterance (DeBERTa-large, 1024-d)
    │       └── manet_UTT/              # visual .npy per utterance (MANet, 512-d)
    ├── IEMOCAPFour/                    # 4-class (IEMOCAPFour)
    │   ├── IEMOCAP_features_raw_4way.pkl
    │   └── features/  (same three subfolders)
    ├── CMUMOSI/
    │   ├── CMUMOSI_features_raw_2way.pkl
    │   └── features/  (same three subfolders)
    └── CMUMOSEI/
        ├── CMUMOSEI_features_raw_2way.pkl
        └── features/  (same three subfolders)
```

Notes:

- The `*_features_raw_*way.pkl` files store utterance IDs / labels / speakers and train–validation–test splits. IEMOCAP uses 151 conversations (Session 5 held out for testing, following SDR-GNN); CMU-MOSI uses the standard 1284/229/686 split; CMU-MOSEI uses 2249/300/676.
- Each `features/<extractor>/` folder holds one `.npy` file per utterance, named by its utterance ID.
- **License:** IEMOCAP, CMU-MOSI and CMU-MOSEI are released under their own academic license agreements that prohibit redistribution. Please obtain them from the official providers:
  - IEMOCAP: https://sail.usc.edu/iemocap/
  - CMU-MOSI / CMU-MOSEI: https://github.com/CMU-MultimodalSDK (SDK & datasets)
  - Feature extraction follows the protocols described in the paper (Wav2Vec 2.0, DeBERTa-large, MANet).
- This repository ships the **empty dataset folder skeleton** (with a README.txt in each folder) so the expected layout is self-explanatory; the `.pkl` and `.npy` data files themselves are not distributed.

## Quick Start

Main training (choose dataset via `--dataset`):

```bash
# IEMOCAP 6-class (5-fold, each Session as test fold)
python train_final_moe.py --dataset IEMOCAPSix  --hidden_dim 512 --dropout 0.45 --save_dir ./results_iemocap6

# IEMOCAP 4-class
python train_final_moe.py --dataset IEMOCAPFour --hidden_dim 512 --dropout 0.45 --save_dir ./results_iemocap4

# CMU-MOSI / CMU-MOSEI (regression, standard split)
python train_final_moe.py --dataset CMUMOSI  --hidden_dim 256 --dropout 0.2 --save_dir ./results_mosi
python train_final_moe.py --dataset CMUMOSEI --hidden_dim 256 --dropout 0.2 --save_dir ./results_mosei
```

Ablation studies (edit `DATASET` / `BEST_PARAMS` at the top, then run):

```bash
python run_ablations.py   # none / no_imputation / no_moe / no_gating / no_hypergraph / no_causal_factor
```

Optional hyperparameter search (edit `TARGET_DATASET` at the top, then run):

```bash
python optuna_search.py
```

Key arguments:

| Arg | Meaning | Default |
|---|---|---|
| `--dataset` | `IEMOCAPSix` / `IEMOCAPFour` / `CMUMOSI` / `CMUMOSEI` | `CMUMOSI` |
| `--folds` | IEMOCAP sessions used as test folds | `1,2,3,4,5` |
| `--hidden_dim` | Latent dim D (512 for IEMOCAP, 256 for MOSI/MOSEI) | 512 |
| `--num_experts` | MoE experts N (paper: N=4, Top-K=2) | 4 |
| `--p1_mask` | Phase-1 training missing rate | 0.4 |
| `--ablation` | Ablation mode (see above) | `none` |

## Outputs

Each run writes to `--save_dir`:

- `train.log` — per-epoch validation metrics
- `fold{X}_phase{1,2,3}.pth` — checkpoints of each training phase
- `cm_data_<dataset>.npy` — saved predictions/labels for confusion-matrix analysis
- Final evaluation reports Accuracy / Weighted-F1 (and MAE for regression) at missing rates η ∈ {0.0, 0.1, ..., 0.7}

## Main Results (from the paper)

- **IEMOCAP (4-class):** 76.06 WAF1 at η=0.7, outperforming SDR-GNN by 1.65 points; at η=0.7 CaM-HG still exceeds most baselines running with complete modalities.
- **IEMOCAP:** +1.43 / +1.25 WAF1 over SOTA on the 4-class / 6-class settings.
- **CMU-MOSI / CMU-MOSEI:** average 81.4 / 84.9 Acc-2 across η ∈ [0, 0.7]; on MOSEI, CaM-HG at η=0.7 (82.6) surpasses strong baselines with complete data.

## Usage Terms

This code is provided for **research use only**. If you find it useful, please cite our paper. You may not redistribute the code or the datasets without permission.

## Citation

```bibtex
@inproceedings{yang2026camhg,
  title     = {CaM-HG: Causal-Enhanced MoE and Hypergraphs Network for Incomplete Multimodal Emotion Recognition in Conversations},
  author    = {Yang, Mingjian and Wang, Yong and Liu, Peng and Yin, Wen},
  booktitle = {Proceedings of ACL},
  year      = {2026}
}
```

## Contact

For questions about the code, please open an issue or contact: Mingjian Yang (ymj0420@std.uestc.edu.cn)
