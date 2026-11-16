import os

# ==============================================================
# 核心修改点：动态获取当前项目根目录
# ==============================================================
# 获取当前 path.py 文件所在的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 定义 dataset 的根目录 (对应你说的 "当前目录下的 dataset")
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')

# 结果保存目录
PATH_TO_SAVE_ROOT = os.path.join(PROJECT_ROOT, 'result')

# ==============================================================
# 路径配置 (Windows/Linux 通用逻辑)
# ==============================================================

# 既然你已经把数据集放在了本地，其实没必要严格区分 Win/Linux 字典
# 这里为了保持你代码结构的一致性，我同时更新了两部分，但逻辑是一样的。

############ For Windows ##############

DATA_DIR_Win = {
    'CMUMOSI': os.path.join(DATASET_ROOT, 'CMUMOSI'),
    'CMUMOSEI': os.path.join(DATASET_ROOT, 'CMUMOSEI'),
    # 假设你的 IEMOCAP 4分类数据在 dataset/IEMOCAPFour
    'IEMOCAPFour': os.path.join(DATASET_ROOT, 'IEMOCAPFour'), 
    # 假设你的 IEMOCAP 6分类数据在 dataset/IEMOCAP
    'IEMOCAPSix': os.path.join(DATASET_ROOT, 'IEMOCAP'),      
}

PATH_TO_FEATURES_Win = {
    'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'features'),
    'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'], 'features'),
    'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'features'),
    'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'features'),
}

PATH_TO_LABEL_Win = {
    'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
    'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
    'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
    'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}

############ For LINUX ##############

DATA_DIR_LINUX = {
    'CMUMOSI': os.path.join(DATASET_ROOT, 'CMUMOSI'),
    'CMUMOSEI': os.path.join(DATASET_ROOT, 'CMUMOSEI'),
    'IEMOCAPSix': os.path.join(DATASET_ROOT, 'IEMOCAP'),
    'IEMOCAPFour': os.path.join(DATASET_ROOT, 'IEMOCAPFour'),
}

PATH_TO_FEATURES_LINUX = {
    'CMUMOSI': os.path.join(DATA_DIR_LINUX['CMUMOSI'], 'features'),
    'CMUMOSEI': os.path.join(DATA_DIR_LINUX['CMUMOSEI'], 'features'),
    'IEMOCAPSix': os.path.join(DATA_DIR_LINUX['IEMOCAPSix'], 'features'),
    'IEMOCAPFour': os.path.join(DATA_DIR_LINUX['IEMOCAPFour'], 'features'),
}

# 注意：你原文件这里写成了 PATH_TO_LABEL_LUNUX (拼写错误)，这里帮你修正为 LINUX
PATH_TO_LABEL_LINUX = {
    'CMUMOSI': os.path.join(DATA_DIR_LINUX['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
    'CMUMOSEI': os.path.join(DATA_DIR_LINUX['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
    'IEMOCAPSix': os.path.join(DATA_DIR_LINUX['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
    'IEMOCAPFour': os.path.join(DATA_DIR_LINUX['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}