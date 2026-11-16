IEMOCAP (6-class, selected via --dataset IEMOCAPSix)
=====================================================

Place in this folder:
  IEMOCAP_features_raw_6way.pkl    (utterance IDs / labels / speakers / splits)

Per-utterance .npy feature files go in:
  features/wav2vec-large-c-UTT/    (audio, Wav2Vec 2.0 large, 1024-d)
  features/deberta-large-4-UTT/    (text, DeBERTa-large, 1024-d)
  features/manet_UTT/              (visual, MANet, 512-d)

NOTICE: Data files are NOT distributed here. IEMOCAP is released under a
license agreement that prohibits redistribution. Please obtain it from
the official provider: https://sail.usc.edu/iemocap/
See the top-level README.md ("Dataset Preparation") for details.
