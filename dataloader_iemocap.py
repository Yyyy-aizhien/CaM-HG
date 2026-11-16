import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def read_data(label_path, feature_root):
    if not isinstance(feature_root, str): return feature_root
    
    # 读取 Label 文件
    try:
        data = pickle.load(open(label_path, "rb"), encoding='latin1')
        if isinstance(data, dict):
             videoIDs, videoLabels, videoSpeakers = data['videoIDs'], data['videoLabels'], data['videoSpeakers']
        else:
             videoIDs, videoLabels, videoSpeakers = data[0], data[1], data[2]
    except Exception as e:
        print(f"Error loading label path: {label_path}\nError: {e}")
        raise

    names = []
    speakers = []
    vids = sorted(list(videoIDs.keys())) 
    
    for vid in vids:
        uids_video = videoIDs[vid]
        spks_video = videoSpeakers[vid]
        names.extend(uids_video)
        speakers.extend(spks_video)

    features = []
    feature_dim = -1
    
    if os.path.exists(feature_root):
        all_files = set(os.listdir(feature_root))
    else:
        print(f"Warning: Directory {feature_root} does not exist.")
        all_files = set()

    iterator = tqdm(enumerate(names), total=len(names), desc=f"Reading {os.path.basename(feature_root)}")
    for ii, name in iterator:
        speaker = speakers[ii]
        # 强制修正 speaker
        if speaker not in ['F', 'M']: speaker = 'M'
            
        target_file = name + ".npy"
        found = False
        feature_path = ""
        
        if target_file in all_files:
            feature_path = os.path.join(feature_root, target_file)
            found = True
        else:
            g_search = glob.glob(os.path.join(feature_root, name + '*'))
            if len(g_search) > 0:
                feature_path = g_search[0]
                found = True
        
        feature = {'F': [], 'M': []}
        if not found:
            single_feature = np.zeros((feature_dim if feature_dim > 0 else 1024,))
            feature[speaker].append(single_feature)
        else:
            if os.path.isdir(feature_path): 
                facenames = sorted(os.listdir(feature_path))
                for facename in facenames:
                    facefeat = np.load(os.path.join(feature_path, facename))
                    feature_dim = max(feature_dim, facefeat.shape[-1])
                    if 'F' in facename: feature['F'].append(facefeat)
                    else: feature['M'].append(facefeat)
            else: 
                single_feature = np.load(feature_path).squeeze()
                feature[speaker].append(single_feature)
                feature_dim = max(feature_dim, single_feature.shape[-1])

            for spk in feature:
                if isinstance(feature[spk], list):
                    if len(feature[spk]) == 0: 
                        single_feature = np.zeros((feature_dim if feature_dim > 0 else 1024, ))
                    else: 
                        single_feature = np.array(feature[spk]).squeeze()
                    if len(single_feature.shape) == 2: 
                        single_feature = np.mean(single_feature, axis=0)
                    feature[spk] = single_feature
        features.append(feature)

    name2feats = {}
    for ii in range(len(names)): name2feats[names[ii]] = features[ii]
    return name2feats, feature_dim

class IEMOCAPDataset(Dataset):
    def __init__(self, label_path, audio_data, text_data, video_data, mask_ratio=0.3, specific_vids=None, task_type='classification'):
        """
        task_type: 'classification' (IEMOCAP) or 'regression' (MOSI/MOSEI)
        """
        # 1. Load Features
        if isinstance(audio_data, str): self.name2audio, self.adim = read_data(label_path, audio_data)
        else: self.name2audio, self.adim = audio_data
        if isinstance(text_data, str): self.name2text, self.tdim = read_data(label_path, text_data)
        else: self.name2text, self.tdim = text_data
        if isinstance(video_data, str): self.name2video, self.vdim = read_data(label_path, video_data)
        else: self.name2video, self.vdim = video_data

        self.mask_ratio = mask_ratio
        self.task_type = task_type
        
        # 2. Load Labels
        data = pickle.load(open(label_path, "rb"), encoding='latin1')
        if isinstance(data, dict):
             self.videoIDs, self.videoLabels, self.videoSpeakers = data['videoIDs'], data['videoLabels'], data['videoSpeakers']
        else:
             self.videoIDs, self.videoLabels, self.videoSpeakers = data[0], data[1], data[2]
        
        # 3. Filter Vids
        if specific_vids is not None:
            self.vids = sorted(specific_vids)
        else:
            self.vids = sorted(list(self.videoIDs.keys()))

        self.data_list = [] 
        speakermap = {'F': 0, 'M': 1} 

        for vid in self.vids:
            if vid not in self.videoIDs: continue

            uids = self.videoIDs[vid]
            labels = self.videoLabels[vid]
            speakers = self.videoSpeakers[vid]
            
            vid_audio, vid_text, vid_video, vid_spk, vid_label = [], [], [], [], []
            
            for ii, uid in enumerate(uids):
                raw_spk = speakers[ii] if ii < len(speakers) else 'M'
                if raw_spk not in ['F', 'M']: raw_spk = 'M'
                
                feat_a = self.name2audio[uid][raw_spk] if isinstance(self.name2audio[uid], dict) else self.name2audio[uid]
                feat_t = self.name2text[uid][raw_spk] if isinstance(self.name2text[uid], dict) else self.name2text[uid]
                feat_v = self.name2video[uid][raw_spk] if isinstance(self.name2video[uid], dict) else self.name2video[uid]
                
                raw_label = labels[ii]
                
                # [关键对齐]
                if self.task_type == 'regression':
                    # MOSI/MOSEI: 保持原始浮点数，保留0值样本用于训练
                    final_label = float(raw_label)
                    # 注意：Regression 任务中，我们这里不丢弃 0 值。
                else:
                    # IEMOCAP: 保持原有离散标签
                    final_label = int(raw_label)

                vid_audio.append(feat_a)
                vid_text.append(feat_t)
                vid_video.append(feat_v)
                vid_spk.append(speakermap[raw_spk])
                vid_label.append(final_label)
            
            if len(vid_label) > 0:
                self.data_list.append({
                    'vid': vid,
                    'audio': torch.FloatTensor(np.array(vid_audio)), 
                    'text': torch.FloatTensor(np.array(vid_text)),   
                    'visual': torch.FloatTensor(np.array(vid_video)),
                    'speakers': torch.LongTensor(np.array(vid_spk)), 
                    # Regression 用 FloatTensor, Classification 用 LongTensor
                    'labels': torch.FloatTensor(np.array(vid_label)) if self.task_type == 'regression' else torch.LongTensor(np.array(vid_label))
                })

    def __getitem__(self, index):
        data = self.data_list[index]
        feat_a = data['audio']
        feat_t = data['text']
        feat_v = data['visual']
        seq_len = feat_a.shape[0]
        
        # Mask
        if self.mask_ratio > 0:
            mask = (torch.rand(seq_len, 3) > self.mask_ratio).float()
            all_zeros = (mask.sum(dim=1) == 0)
            if all_zeros.any():
                indices = torch.randint(0, 3, (all_zeros.sum(),))
                mask[all_zeros, indices] = 1.0
        else:
            mask = torch.ones(seq_len, 3)
            
        return {
            'audio_gt': feat_a, 'text_gt': feat_t, 'visual_gt': feat_v,
            'audio_masked': feat_a * mask[:, 0:1], 
            'text_masked': feat_t * mask[:, 1:2], 
            'visual_masked': feat_v * mask[:, 2:3],
            'mask': mask, 
            'speakers': data['speakers'], 'labels': data['labels'], 'vid': data['vid']
        }

    def __len__(self): return len(self.data_list)

def collate_fn(batch):
    batch_data = {}
    pad_keys = ['audio_gt', 'text_gt', 'visual_gt', 'audio_masked', 'text_masked', 'visual_masked', 'mask', 'speakers', 'labels']
    for key in pad_keys:
        tensors = [item[key] for item in batch]
        
        # [关键修正] 对于 Regression (float labels)，padding value 设为 0 (或者一个极值)，计算 Loss 时必须 mask 掉
        # 这里为了安全，对于 Labels，如果不确定类型，先按 padding_value=0 处理，后续在 Loss 中处理 Mask
        if key == 'labels':
            if tensors[0].dtype == torch.float32: # Regression
                pv = -100.0 # 随便设一个标记值，但计算 Loss 时必须用 mask 忽略
            else: # Classification
                pv = -1 
        else:
            pv = 0
            
        batch_data[key] = pad_sequence(tensors, batch_first=True, padding_value=pv)
        
    batch_data['lengths'] = torch.LongTensor([len(item['labels']) for item in batch])
    batch_data['vid'] = [item['vid'] for item in batch]
    return batch_data