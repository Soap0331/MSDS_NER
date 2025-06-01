import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torchcrf import CRF

# 设置随机种子以确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 自定义模型输出
class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss

# BERT-CRF模型定义（简化版，无BiLSTM）
class BertCrf(nn.Module):
    def __init__(self, bert_model_name, num_labels, dropout_rate=0.3):
        super(BertCrf, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_config = BertConfig.from_pretrained(bert_model_name)
        hidden_size = self.bert_config.hidden_size

        # Dropout层（增加正则化）
        self.dropout = nn.Dropout(dropout_rate)

        # 线性层
        self.classifier = nn.Linear(hidden_size, num_labels)

        # CRF层
        self.crf = CRF(num_labels, batch_first=True)

        # 初始化线性层权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # BERT层
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 获取序列输出
        sequence_output = bert_output[0]  # [batch_size, seq_len, hidden_size]

        # Dropout
        sequence_output = self.dropout(sequence_output)

        # 线性层
        emissions = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        # 准备掩码（CRF需要布尔掩码）
        mask = attention_mask.bool()

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 将-100替换为0（仅用于CRF计算，不影响实际损失）
            valid_labels = labels.clone()
            valid_labels[labels == -100] = 0

            # 对于CRF，我们需要确保第一个时间步的掩码为True
            # 但同时需要考虑标签掩码
            label_mask = (labels != -100)

            # 使用attention_mask作为CRF的掩码，但要确保有效标签位置
            # CRF期望mask的第一个时间步必须为True
            crf_mask = mask.clone()

            # 如果某个序列的所有标签都是-100，至少保留第一个位置
            batch_size = crf_mask.size(0)
            for i in range(batch_size):
                if not label_mask[i].any():
                    # 如果整个序列都没有有效标签，至少保留第一个位置
                    crf_mask[i, 0] = True

            # 计算CRF损失
            log_likelihood = self.crf(emissions, valid_labels, mask=crf_mask, reduction='mean')
            loss = -log_likelihood

        # 预测（维特比解码）
        predictions = self.crf.decode(emissions, mask=mask)

        return ModelOutput(predictions, labels, loss)

# 数据加载函数
def load_bio_data(file_path):
    """加载BIO格式的NER数据"""
    sentences = []
    labels = []

    sentence = []
    label = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[-1]
                    sentence.append(word)
                    label.append(tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

# 加载CoNLL格式的数据
def load_conll_data(file_path):
    """加载CoNLL格式的NER数据"""
    sentences = []
    labels = []

    sentence = []
    label = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("-DOCSTART-"):
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[-1]
                    sentence.append(word)
                    label.append(tag)
            elif sentence:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

# 创建标签映射
def create_tag_mapping(labels):
    """创建标签到ID的映射和ID到标签的映射"""
    unique_tags = set()
    for seq in labels:
        for tag in seq:
            unique_tags.add(tag)

    # 确保'O'标签总是索引0
    ordered_tags = ['O'] + [tag for tag in sorted(list(unique_tags)) if tag != 'O']

    tag2id = {tag: i for i, tag in enumerate(ordered_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    return tag2id, id2tag

# 自定义数据集类
class EnglishNERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, tag2id, max_len=128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]

        # 使用FastTokenizer对单词进行分词
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # 获取输入ID、注意力掩码和标记类型ID
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        token_type_ids = encoding['token_type_ids'].squeeze()

        # 将标签映射到输入标记
        word_ids = encoding.word_ids()

        # 创建标签ID列表，使用模型忽略的索引-100作为填充
        label_ids = torch.ones(self.max_len, dtype=torch.long) * -100

        # 映射标签：只标记每个单词的第一个子词
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                # 对于特殊标记（[CLS], [SEP], [PAD]），我们需要特殊处理
                if i == 0:  # [CLS] token
                    # 给[CLS]标记分配'O'标签（假设是0）
                    label_ids[i] = self.tag2id.get('O', 0)
                else:
                    label_ids[i] = -100
            elif word_idx != previous_word_idx:
                # 只对每个单词的第一个标记分配标签
                label_ids[i] = self.tag2id[tags[word_idx]]
            else:
                # 对同一单词的后续标记使用-100
                label_ids[i] = -100
            previous_word_idx = word_idx

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': label_ids
        }

# 训练函数
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, id2tag, tag2id, epochs=3, patience=3, output_dir='./model_output'):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model.pt')

    # 记录训练损失和验证F1分数
    training_stats = {
        'train_loss': [],
        'val_f1': [],
        'epochs': []
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in progress_bar:
            # 将输入数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # 清除上一轮的梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs.loss
            epoch_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()
            scheduler.step()

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

        # 计算每个epoch的平均损失
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        training_stats['train_loss'].append(avg_epoch_loss)
        training_stats['epochs'].append(epoch + 1)

        print(f"Average training loss: {avg_epoch_loss:.4f}")

        # 在验证集上评估模型
        print("Validating model...")
        val_f1, val_loss = evaluate(model, val_dataloader, device, id2tag)
        training_stats['val_f1'].append(val_f1)

        print(f"Validation F1 score: {val_f1:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        # 检查是否是最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            print(f"Found new best model! F1: {val_f1:.4f}")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"Model not improved. Patience: {patience_counter}/{patience}")

        # 检查早停条件
        if patience_counter >= patience:
            print(f"Early stopping triggered! Best validation F1: {best_f1:.4f}")
            break

    # 加载最佳模型
    if os.path.exists(best_model_path):
        print("Loading best model weights...")
        model.load_state_dict(torch.load(best_model_path))

    # 可视化训练过程
    plot_training_stats(training_stats, output_dir)

    return model, training_stats

# 评估函数
def evaluate(model, eval_dataloader, device, id2tag):
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # 将输入数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # 不计算梯度
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            if outputs.loss is not None:
                total_loss += outputs.loss.item()

        # 获取预测结果（已经是解码后的标签索引列表）
        predictions = outputs.logits

        # 处理每个批次的样本
        for i in range(len(predictions)):
            pred_tags = []
            true_tags = []

            # CRF解码结果的长度可能与原始序列长度不同
            # 我们需要根据attention_mask来对齐
            pred_seq = predictions[i]
            pred_idx = 0

            for j in range(attention_mask.size(1)):
                if attention_mask[i, j] == 1:
                    # 这是一个有效位置
                    if labels[i, j] != -100:
                        # 这个位置有有效标签
                        if pred_idx < len(pred_seq):
                            pred_id = pred_seq[pred_idx]
                            true_id = labels[i, j].item()

                            pred_tags.append(id2tag[pred_id])
                            true_tags.append(id2tag[true_id])

                    pred_idx += 1

            if pred_tags and true_tags:
                all_predictions.append(pred_tags)
                all_labels.append(true_tags)

    # 计算F1分数
    f1 = f1_score(all_labels, all_predictions)
    avg_loss = total_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0

    return f1, avg_loss

# 完整的评估函数（返回详细报告）
def evaluate_detailed(model, eval_dataloader, device, id2tag):
    model.eval()

    true_labels = []
    pred_labels = []
    true_label_sequences = []
    pred_label_sequences = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # 将输入数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # 不计算梯度
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # 获取预测结果
        predictions = outputs.logits

        # 处理每个批次的样本
        for i in range(len(predictions)):
            sample_labels = []
            sample_preds = []

            # CRF解码结果的长度可能与原始序列长度不同
            pred_seq = predictions[i]
            pred_idx = 0

            # 遍历整个序列
            for j in range(attention_mask.size(1)):
                if attention_mask[i, j] == 1:
                    # 这是一个有效位置
                    if labels[i, j] != -100:
                        # 这个位置有有效标签
                        if pred_idx < len(pred_seq):
                            pred_id = pred_seq[pred_idx]
                            true_id = labels[i, j].item()

                            true_labels.append(true_id)
                            pred_labels.append(pred_id)

                            sample_labels.append(id2tag[true_id])
                            sample_preds.append(id2tag[pred_id])

                    pred_idx += 1

            if sample_labels:
                true_label_sequences.append(sample_labels)
                pred_label_sequences.append(sample_preds)

    # 转换ID为标签
    true_tags = [id2tag[label_id] for label_id in true_labels]
    pred_tags = [id2tag[pred_id] for pred_id in pred_labels]

    # 计算标记级别的分类报告
    token_report = classification_report(true_tags, pred_tags, digits=4)

    # 计算实体级别的分类报告
    entity_report = seq_classification_report(true_label_sequences, pred_label_sequences, digits=4)

    return token_report, entity_report, true_tags, pred_tags, true_label_sequences, pred_label_sequences

# 可视化训练过程
def plot_training_stats(stats, save_path):
    plt.figure(figsize=(12, 5))

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制训练损失
    ax1.plot(stats['epochs'], stats['train_loss'], 'b-o', label='Training Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制验证F1分数
    ax2.plot(stats['epochs'], stats['val_f1'], 'r-o', label='Validation F1')
    ax2.set_title('Validation F1 over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_stats.png'))
    plt.close()

# 可视化混淆矩阵
def plot_confusion_matrix(true_tags, pred_tags, save_path):
    # 获取标签的唯一值
    unique_tags = sorted(list(set(true_tags + pred_tags)))

    # 创建混淆矩阵
    cm = np.zeros((len(unique_tags), len(unique_tags)), dtype=int)
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}

    for true_tag, pred_tag in zip(true_tags, pred_tags):
        cm[tag_to_idx[true_tag]][tag_to_idx[pred_tag]] += 1

    # 绘制热图
    plt.figure(figsize=(12, 10))

    if len(unique_tags) > 30:
        sns.heatmap(cm, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (labels omitted due to size)')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_tags, yticklabels=unique_tags)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

# 预测函数
def predict_sentence(model, tokenizer, sentence, tag2id, id2tag, device, max_len=128):
    model.eval()

    # 处理句子
    if isinstance(sentence, str):
        tokens = sentence.split()
    else:
        tokens = sentence

    # 对tokens进行编码
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors='pt',
        max_length=max_len,
        padding='max_length',
        truncation=True
    )

    # 将输入数据移动到设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    # 不计算梯度
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # 获取预测结果（CRF解码后的结果）
    predictions = outputs.logits[0]  # 第一个样本

    # 获取word_ids映射
    word_ids = encoding.word_ids()

    # 初始化结果
    entities = {}
    current_entity = {'type': None, 'start': -1, 'end': -1, 'text': []}

    # 遍历预测结果
    pred_idx = 0
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and attention_mask[0, i] == 1:
            # 只处理第一个子词
            if i == 0 or word_ids[i-1] != word_idx:
                if pred_idx < len(predictions):
                    pred_id = predictions[pred_idx]
                    pred_idx += 1

                    if pred_id in id2tag:
                        tag = id2tag[pred_id]

                        # 如果是B-开头，标记新实体开始
                        if tag.startswith('B-'):
                            # 保存之前的实体
                            if current_entity['type'] is not None:
                                ent_type = current_entity['type']
                                if ent_type not in entities:
                                    entities[ent_type] = []
                                entity_text = ' '.join(current_entity['text'])
                                entities[ent_type].append((
                                    entity_text,
                                    current_entity['start'],
                                    current_entity['end']
                                ))

                            # 开始新实体
                            entity_type = tag[2:]
                            current_entity = {
                                'type': entity_type,
                                'start': word_idx,
                                'end': word_idx,
                                'text': [tokens[word_idx]]
                            }

                        # 如果是I-开头，继续当前实体
                        elif tag.startswith('I-'):
                            entity_type = tag[2:]
                            if current_entity['type'] == entity_type:
                                current_entity['end'] = word_idx
                                current_entity['text'].append(tokens[word_idx])

                        # 如果是O，结束当前实体
                        elif tag == 'O':
                            if current_entity['type'] is not None:
                                ent_type = current_entity['type']
                                if ent_type not in entities:
                                    entities[ent_type] = []
                                entity_text = ' '.join(current_entity['text'])
                                entities[ent_type].append((
                                    entity_text,
                                    current_entity['start'],
                                    current_entity['end']
                                ))
                                current_entity = {'type': None, 'start': -1, 'end': -1, 'text': []}

    # 处理最后一个实体
    if current_entity['type'] is not None:
        ent_type = current_entity['type']
        if ent_type not in entities:
            entities[ent_type] = []
        entity_text = ' '.join(current_entity['text'])
        entities[ent_type].append((
            entity_text,
            current_entity['start'],
            current_entity['end']
        ))

    return entities

def main():
    parser = argparse.ArgumentParser(description='Train BERT-CRF NER model for English text')
    parser.add_argument('--data_path', type=str, required=True, help='Path to BIO/CoNLL formatted data file')
    parser.add_argument('--data_format', type=str, default='conll', choices=['bio', 'conll'], help='Data format: bio or conll')
    parser.add_argument('--output_dir', type=str, default='bert_crf_english_model', help='Directory to save model and results')
    parser.add_argument('--bert_model', type=str, default='bert-base-cased', help='BERT model to use')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5, help='Learning rate for BERT layers')
    parser.add_argument('--crf_learning_rate', type=float, default=5e-4, help='Learning rate for CRF layer')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training for lr warmup')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据
    print("Loading data...")
    if args.data_format == 'bio':
        sentences, labels = load_bio_data(args.data_path)
    else:
        sentences, labels = load_conll_data(args.data_path)

    print(f"Loaded {len(sentences)} sentences")

    # 创建标签映射
    tag2id, id2tag = create_tag_mapping(labels)
    print(f"Unique tags: {list(tag2id.keys())}")

    # 将标签映射保存到文件
    with open(os.path.join(args.output_dir, "tag_mapping.json"), "w") as f:
        json.dump({"tag2id": tag2id, "id2tag": id2tag}, f)

    # 划分训练集和测试集
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=args.test_size, random_state=args.seed
    )

    print(f"Training set size: {len(train_sentences)}")
    print(f"Validation set size: {len(val_sentences)}")

    # 加载BERT分词器
    print(f"Loading {args.bert_model} tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)

    # 创建数据集
    print("Preparing datasets...")
    train_dataset = EnglishNERDataset(train_sentences, train_labels, tokenizer, tag2id, max_len=args.max_len)
    val_dataset = EnglishNERDataset(val_sentences, val_labels, tokenizer, tag2id, max_len=args.max_len)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 加载BERT-CRF模型
    print(f"Building BERT-CRF model...")
    model = BertCrf(args.bert_model, len(tag2id), dropout_rate=args.dropout_rate)
    model.to(device)

    # 计算总训练步数
    total_steps = len(train_dataloader) * args.epochs

    # 设置优化器和学习率调度器（使用分层学习率）
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert':
            bert_param_optimizer.append((name, para))
        else:
            crf_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # BERT参数
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},

        # CRF参数（使用较低的学习率）
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps
    )

    # 训练模型
    print("Starting model training...")
    model, training_stats = train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device,
        id2tag,
        tag2id,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )

    # 最终评估模型
    print("Final model evaluation...")
    token_report, entity_report, true_tags, pred_tags, true_label_sequences, pred_label_sequences = evaluate_detailed(model, val_dataloader, device, id2tag)

    print("Token-level classification report:")
    print(token_report)

    print("\nEntity-level classification report:")
    print(entity_report)

    # 保存评估报告
    with open(os.path.join(args.output_dir, "evaluation_report.txt"), "w") as f:
        f.write("Token-level classification report:\n")
        f.write(token_report)
        f.write("\n\nEntity-level classification report:\n")
        f.write(entity_report)

    # 可视化混淆矩阵
    plot_confusion_matrix(true_tags, pred_tags, args.output_dir)

    # 保存模型
    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)

    # 保存模型配置
    model_config = {
        "bert_model": args.bert_model,
        "max_len": args.max_len,
        "num_labels": len(tag2id),
        "tag2id": tag2id,
        "id2tag": id2tag,
        "dropout_rate": args.dropout_rate,
    }

    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    print("Model training completed!")

    # 示例预测
    print("\nExample predictions:")

    test_sentences = val_sentences[:10]  # 使用验证集的前三个句子
    for test_sentence in test_sentences:
        print(f"\nInput: {' '.join(test_sentence)}")
        entities = predict_sentence(model, tokenizer, test_sentence, tag2id, id2tag, device, max_len=args.max_len)
        print("Prediction results:")
        for entity_type, entity_list in entities.items():
            print(f"{entity_type}: {entity_list}")


# 英文NER预测器类
class EnglishNERPredictor:
    def __init__(self, model_path, device=None):
        """
        初始化英文NER预测器
        Args:
            model_path: 模型保存路径
            device: 设备（CPU或GPU）
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # 加载模型配置
        with open(os.path.join(model_path, "model_config.json"), "r") as f:
            config = json.load(f)

        self.max_len = config["max_len"]
        self.tag2id = config["tag2id"]
        self.id2tag = {int(k): v for k, v in config["id2tag"].items()}
        self.dropout_rate = config.get("dropout_rate", 0.1)

        # 加载分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)

        # 加载模型
        self.model = BertCrf(
            config["bert_model"],
            len(self.tag2id),
            dropout_rate=self.dropout_rate
        )

        # 加载模型权重
        self.model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def predict(self, text):
        """
        预测文本中的实体
        Args:
            text: 文本字符串或标记列表
        Returns:
            包含实体的字典，格式为 {实体类型: [(实体文本, 开始位置, 结束位置)]}
        """
        # 处理输入文本
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text

        return predict_sentence(
            self.model,
            self.tokenizer,
            tokens,
            self.tag2id,
            self.id2tag,
            self.device,
            max_len=self.max_len
        )

    def predict_batch(self, texts):
        """
        批量预测文本中的实体
        Args:
            texts: 文本列表
        Returns:
            实体列表
        """
        results = []
        for text in texts:
            entities = self.predict(text)
            results.append(entities)
        return results


if __name__ == "__main__":
    main()

    # 使用示例：
    # python bert_crf_train.py --data_path output0518_enhance.bio --data_format conll --output_dir bert_crf_model0519

    # 预测示例：
    # predictor = EnglishNERPredictor("bert_crf_model")
    # text = "John Smith works at Microsoft in Seattle."
    # entities = predictor.predict(text)
    # print(entities)