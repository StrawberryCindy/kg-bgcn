import os
import json
import torch
from tqdm import tqdm
from torch import nn
from transformers import AutoTokenizer


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_labels, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

    def hidden_forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x, (hn, cn) = self.lstm(x)
        return x[:, -1, :]


    # 定义批次嵌入提取函数
def get_cls_embeddings(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        cls_embeddings = model.hidden_forward(input_ids=input_ids, attention_mask=attention_mask)
    return cls_embeddings

def encode_text_files(model, tokenizer, file_dir):
    file_list = os.listdir(file_dir)
    for file_name in tqdm(file_list, desc='Encoding files'):
        if not file_name.endswith('pkl.json'):
            continue
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text_list = [item['text'] for item in data]
        lable = data[0]['label']
        cls_embeddings = get_cls_embeddings(model, tokenizer, text_list)
        info = {
            'label': lable,
            'feature': cls_embeddings.tolist()
        }
        save_name = file_name.replace('.json', '_encoded.json')
        save_path = os.path.join(file_dir, save_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    file_dir = 'data/data_lstm_kg'
    train_dir = os.path.join(file_dir, 'train')
    val_dir = os.path.join(file_dir, 'val')
    test_dir = os.path.join(file_dir, 'test')
    # 加载预训练的lstm模型和分词器
    # 可以使用自己的模型和分词器
    model_name = 'models/results_20240819_072900/best_model'
    model_path = os.path.join(model_name, 'pytorch_model.bin')

    num_labels = 15
    # 初始化Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    model = LSTMClassifier(vocab_size=vocab_size, embed_size=128, hidden_size=768, num_labels=num_labels, num_layers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 加载模型权重,如果想要未训练的，请注释掉
    model.load_state_dict(torch.load(model_path))

    # 设置模型为评估模式
    model.eval()

    encode_text_files(model, tokenizer, train_dir)
    encode_text_files(model, tokenizer, val_dir)
    encode_text_files(model, tokenizer, test_dir)
