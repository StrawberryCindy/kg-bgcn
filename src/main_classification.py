import json
import torch
import datasets
from data_utils import load_and_prepare_datasets, load_and_prepare_datasets_shuffle
from transformers import Trainer, AutoTokenizer, ErnieModel, BertForMaskedLM, TrainingArguments, BertForSequenceClassification, BertTokenizer
from model_utils import (
    initialize_classification_lm_model_and_tokenizer,
    get_compute_accuracy_fn,
    setup_training_args,
    create_output_directory,
    save_config
)
from plot_utils import save_losses, plot_and_save_losses, plot_and_save_accuracies
import os
import shutil
from log_utils import log_info


def main(data_path):
    # 加载配置文件
    print(os.getcwd())
    with open('./src/config.json', 'r') as f:
        config = json.load(f)

    # 获取用户输入的模型类型
    classification_type = config.get("classification_type", "device")

    # 检查分类类型是否有效
    if classification_type not in ["device", "factory", "node"]:
        raise ValueError(
            "Invalid classification type. Choose 'device' or 'factory'.")
    else:
        print(f"Using classification type: {classification_type}")

    model_name = config["model_name_classification"]
    # 更新配置中的模型类型
    config['classification_type'] = classification_type

    # 初始化Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 创建输出目录
    output_dir = create_output_directory()

    log_file = os.path.join(output_dir, 'log.txt')
    log_info(log_file, f"Using classification type: {classification_type}")

    data_dir = os.path.join(output_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 保存配置文件
    save_config(output_dir, config)

    # 动态选择 compute_accuracy 函数
    compute_accuracy = get_compute_accuracy_fn(config['classification_type'])

    # 定义 compute_metrics 闭包
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        accuracy = compute_accuracy(logits, labels)
        return {"accuracy": accuracy}

    # 选择数据文件路径
    # 选择数据文件路径
    if classification_type == "device":
        data_file = data_path + '/device.json'
    if classification_type == "factory":
        data_file = data_path + '/factory.json'
    if classification_type == "node":
        data_dir = data_path
        tokenized_datasets = load_and_prepare_datasets(
            data_dir, tokenizer, config['max_length'])
    else:
        # 加载和准备数据集
        tokenized_datasets = load_and_prepare_datasets_shuffle(
            data_file, tokenizer, config['max_length'], data_dir)

    num_labels = config['num_labels']

    # 初始化Tokenizer和模型
    model, tokenizer = initialize_classification_lm_model_and_tokenizer(
        model_name, num_labels)

    # 设置训练参数
    training_args = setup_training_args(output_dir, config)
    
    device = torch.device('mps')
    model.to(device)

    # 保存训练和验证的损失和准确率
    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics
    )

    # 自定义训练过程，记录损失
    # for epoch in range(int(training_args.num_train_epochs)):
        # print(f"Epoch {epoch+1}/{training_args.num_train_epochs}")
        # log_info(log_file, f"Epoch {epoch+1}/{training_args.num_train_epochs}")
    train_result = trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best_model'))
    train_losses.append(train_result.training_loss)

    # 使用 predict 方法在训练集上预测，计算准确率
    train_predictions = trainer.predict(tokenized_datasets['train'])
    train_accuracy = compute_metrics(
        (train_predictions.predictions, train_predictions.label_ids))["accuracy"]
    train_accuracies.append(train_accuracy)

    eval_result = trainer.evaluate()
    eval_losses.append(eval_result['eval_loss'])
    eval_accuracies.append(eval_result['eval_accuracy'])

    # 评估模型
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    print(f"Test accuracy: {eval_results['eval_accuracy']}")
    log_info(log_file, f"Test accuracy: {eval_results['eval_accuracy']}")

    # 保存模型
    best_model_dir = os.path.join(output_dir, 'best_model')
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print("Model saved successfully to best_model.")
    log_info(log_file, "Model saved successfully to best_model.")

    # 删除 latest_classification_model 文件夹中的内容
    latest_model_dir = 'latest_classification_model'
    if os.path.exists(latest_model_dir):
        shutil.rmtree(latest_model_dir)

    # 将最佳模型保存到 latest_classification_model 文件夹中
    shutil.copytree(best_model_dir, latest_model_dir)

    # 保存损失数据到文件
    save_losses(output_dir, train_losses, eval_losses)

    # 保存准确率数据到文件并绘制图像
    plot_and_save_accuracies(output_dir, train_accuracies, eval_accuracies)

    # 绘制并保存训练和验证的损失图
    plot_and_save_losses(output_dir, train_losses, eval_losses)


if __name__ == "__main__":
    # data_path = r'E:\asset_kg\data_sample\graph_text\data1014'
    # data_path = r'E:\asset_kg\data_sample\graph_text\sample1014\data_factory_node'
    # data_path = r'E:\asset_kg\wangan_code\classification\results\results_20241217_152555\data_node'
    
    # config 此时修改"classification_type":"factory"/"device"
    # 此时data_path 为factory.json所在文件夹
    # num_labels 记得也要改一下
    status_key = '0b8744fe-550c-4d3f-b450-ffbf65455d28'
    data_dir = f"./data_process/train_data/{status_key}"
    main(data_dir)
