import os
import json
import math
import random
from datasets import Dataset, DatasetDict, load_dataset


def load_and_prepare_datasets(data_dir, tokenizer, max_length):
    # 加载数据集
    with open(os.path.join(data_dir, 'train.json'), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, 'val.json'), 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(os.path.join(data_dir, 'test.json'), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    # train_data = train_data[:100]
    # val_data = val_data[:10]
    # test_data = test_data[:10]
    # 转换为 Hugging Face 数据集格式
    train_dataset = Dataset.from_dict({"text": [item['text'] for item in train_data], "label": [
                                      item['label'] for item in train_data]})
    val_dataset = Dataset.from_dict({"text": [item['text'] for item in val_data], "label": [
                                    item['label'] for item in val_data]})
    test_dataset = Dataset.from_dict({"text": [item['text'] for item in test_data], "label": [
                                     item['label'] for item in test_data]})

    dataset = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset})

    # Tokenize函数
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    # 应用tokenize函数
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(
        lambda examples: {'labels': examples['label']}, batched=True)
    tokenized_datasets.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_datasets


def load_and_prepare_datasets_shuffle(data_file, tokenizer, max_length, data_dir):
    # 加载数据集
    with open(data_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    # json_data = json_data[:50]
    train_data, val_data, test_data = shuffle_dataset(json_data)
    with open(os.path.join(data_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(data_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(data_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    # 转换为 Hugging Face 数据集格式
    train_dataset = Dataset.from_dict({"text": [item['text'] for item in train_data], "label": [
                                      item['label'] for item in train_data]})
    val_dataset = Dataset.from_dict({"text": [item['text'] for item in val_data], "label": [
                                    item['label'] for item in val_data]})
    test_dataset = Dataset.from_dict({"text": [item['text'] for item in test_data], "label": [
                                     item['label'] for item in test_data]})

    dataset = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset})

    # Tokenize函数
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    # 应用tokenize函数
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(
        lambda examples: {'labels': examples['label']}, batched=True)
    tokenized_datasets.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_datasets


def shuffle_dataset(data_list):
    lable_dict = {}
    for data in data_list:
        lable = data['label']
        if lable in lable_dict:
            lable_dict[lable].append(data)
        else:
            lable_dict[lable] = [data]
    train_data = []
    test_data = []
    val_data = []
    for lable, data in lable_dict.items():
        random.shuffle(data)
        t1 = math.ceil(len(data)*0.8)
        t2 = math.ceil(len(data)*0.9)
        train_data.extend(data[:t1])
        val_data.extend(data[t1:t2])
        test_data.extend(data[t2:])
    return train_data, val_data, test_data


def balance_data(data_list):
    lable_dict = {}
    for data in data_list:
        lable = data['label']
        if lable in lable_dict:
            lable_dict[lable].append(data)
        else:
            lable_dict[lable] = [data]
    new_data = []
    for lable, data in lable_dict.items():
        if len(data) > 10:
            # length = len(data) % 5 + 5
            length = 10
            new_data.extend(data[:length])
        else:
            length = len(data)
            new_data.extend(data)
        print(lable, len(data), length)
    return new_data

if __name__ == '__main__':
    # data_file = 'results/results_20240813_171727/data/train.json'
    # with open(data_file, 'r', encoding='utf-8') as f:
    #     json_data = json.load(f)
    # balance_data(json_data)
    # print('==================================')
    # data_file = 'results/results_20240813_171727/data/val.json'
    # with open(data_file, 'r', encoding='utf-8') as f:
    #     json_data = json.load(f)
    # balance_data(json_data)
    # print('==================================')
    result_lable = 'factory_results_20240822_054736'
    data_file = f'results/{result_lable}/data/test.json'
    save_file = f'results/{result_lable}/data/test_ba.json'
    with open(data_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    new_data = balance_data(json_data)
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
