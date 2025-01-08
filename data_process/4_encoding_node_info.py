import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# 加载预训练的BERT模型和分词器
model_name = 'models/bert-classification'
# model_name = 'results\results_20240813_080040\best_model'  # 可以替换为你自己的模型
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 设置模型为评估模式
model.eval()

# 定义批次嵌入提取函数
def get_cls_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
    
    # 提取 [CLS] 位置的嵌入向量，CLS 位置在输入序列的第一个位置
    cls_embeddings = last_hidden_state[:, 0, :]
    
    return cls_embeddings

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
        # cls_embeddings = get_cls_embeddings(text_list)
        features = np.random.rand(len(text_list), 768)  # 生成随机特征向量
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