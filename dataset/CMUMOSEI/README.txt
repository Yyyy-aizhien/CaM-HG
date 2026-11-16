CMU-MOSEI (selected via --dataset CMUMOSEI)
============================================

Place in this folder:
  CMUMOSEI_features_raw_2way.pkl   (utterance IDs / labels / speakers / splits)

Per-utterance .npy feature files go in:
  features/wav2vec-large-c-UTT/    (audio, Wav2Vec 2.0 large, 1024-d)
  features/deberta-large-4-UTT/    (text, DeBERTa-large, 1024-d)
  features/manet_UTT/              (visual, MANet, 512-d)

NOTICE: Data files are NOT distributed here. CMU-MOSEI is released under an
academic license that prohibits redistribution. Please obtain it from the
official provider: https://github.com/CMU-MultimodalSDK
See the top-level README.md ("Dataset Preparation") for details.
