import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def encode_text_files(file_dir):
    file_list = os.listdir(file_dir)
    for file_name in tqdm(file_list, desc='Encoding files'):
        if not file_name.endswith('pkl.json'):
            continue
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text_list = [item['text'] for item in data]
        lable = data[0]['label']
        features = np.random.rand(len(text_list), 768)  # 生成随机特征向量
        # 随机正负号
        features = np.where(np.random.rand(*features.shape) < 0.5, -features, features)
        info = {
            'label': lable,
            'feature': features.tolist()
        }
        save_name = file_name.replace('.json', '_encoded.json')
        save_path = os.path.join(file_dir, save_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    file_dir = 'data/data_random'
    train_dir = os.path.join(file_dir, 'train')
    val_dir = os.path.join(file_dir, 'val')
    test_dir = os.path.join(file_dir, 'test')

    encode_text_files(train_dir)
    encode_text_files(val_dir)
    encode_text_files(test_dir)
    # encode_text_files(r'D:\SX\device_graph_3\sample6\data\factory')