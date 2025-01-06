import torch
from transformers import AutoTokenizer,  BertForSequenceClassification,  Trainer, TrainingArguments
from datetime import datetime
import os
import json

def initialize_classification_lm_model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


def get_compute_accuracy_fn(classification_type):
    if classification_type == "mask":
        def compute_accuracy(predictions, labels):
            preds = torch.argmax(predictions, dim=2)
            mask = (labels != -100)
            correct = (preds == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            return accuracy
    else:
        def compute_accuracy(predictions, labels):
            preds = torch.argmax(predictions, dim=1)
            accuracy = (preds == labels).float().mean().item()
            return accuracy
    return compute_accuracy


def setup_training_args(output_dir, config):
    return TrainingArguments(
        output_dir=os.path.join(output_dir, 'best_model'),
        evaluation_strategy=config['evaluation_strategy'],
        save_strategy=config['save_strategy'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        num_train_epochs=config['num_train_epochs'],
        weight_decay=config['weight_decay'],
        logging_dir=os.path.join(output_dir, 'logs'),
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        save_total_limit=config['save_total_limit'],
        logging_steps=10,
        report_to="none"
    )

def create_output_directory():
    main_output_dir = './results'
    os.makedirs(main_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(main_output_dir, f'results_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_config(output_dir, config):
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)