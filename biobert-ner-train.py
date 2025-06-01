import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 设置随机种子以确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 数据加载函数
def load_data(file_path):
    """
    加载BIO格式的NER数据
    返回句子列表和对应的标签列表
    """
    sentences = []
    labels = []

    sentence = []
    label = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 按制表符拆分单词和标签
                parts = line.split('\t')
                if len(parts) == 2:
                    word, tag = parts
                    sentence.append(word)
                    label.append(tag)
            else:
                if sentence:  # 确保句子不为空
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []

    # 确保最后一个句子被添加
    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

# 创建标签映射
def create_tag_mapping(labels):
    """
    创建标签到ID的映射和ID到标签的映射
    """
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
class NERDataset(Dataset):
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

        # 对单词进行分词
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
        token_type_ids = encoding.get('token_type_ids', None)  # 某些模型可能没有token_type_ids

        try:
            # 将标签映射到输入标记
            word_ids = encoding.word_ids()

            # 创建标签ID列表，使用模型忽略的索引-100作为填充
            label_ids = torch.ones(self.max_len, dtype=torch.long) * -100

            # 标记每个单词的第一个子词
            previous_word_idx = None
            for i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    label_ids[i] = -100
                elif word_idx != previous_word_idx:
                    # 只对每个单词的第一个标记分配标签
                    label_ids[i] = self.tag2id[tags[word_idx]]
                else:
                    # 对同一单词的后续标记使用-100
                    label_ids[i] = -100
                previous_word_idx = word_idx

        except (AttributeError, ValueError) as e:
            print(f"处理索引{idx}的样本时出错: {e}")
            raise

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }

        # 只有在模型需要token_type_ids时才添加
        if token_type_ids is not None:
            result['token_type_ids'] = token_type_ids.squeeze()

        return result

# 计算实体级F1分数
def compute_entity_f1(true_tags, pred_tags):
    """从预测和真实标签序列计算实体级F1分数"""
    # 转换为实体级标签序列
    true_entities = []
    pred_entities = []

    for true_seq, pred_seq in zip(true_tags, pred_tags):
        true_entities.append(true_seq)
        pred_entities.append(pred_seq)

    return f1_score(true_entities, pred_entities)

# 训练函数
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, id2tag, tag2id, epochs=3, patience=3):
    best_f1 = 0
    patience_counter = 0
    best_model_path = 'best_model.pt'

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
            labels = batch['labels'].to(device)

            # 检查是否有token_type_ids
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # 清除上一轮的梯度
            optimizer.zero_grad()

            # 构建模型输入
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

            # 只有在模型需要token_type_ids时才添加
            if token_type_ids is not None and 'token_type_ids' in model.forward.__code__.co_varnames:
                model_inputs['token_type_ids'] = token_type_ids

            # 前向传播
            outputs = model(**model_inputs)

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
        print("正在验证模型...")
        token_report, entity_report, true_tags, pred_tags, true_label_sequences, pred_label_sequences = evaluate(model, val_dataloader, device, id2tag, tag2id)
        val_f1 = compute_entity_f1(true_label_sequences, pred_label_sequences)
        training_stats['val_f1'].append(val_f1)

        print(f"验证集F1分数: {val_f1:.4f}")

        # 检查是否是最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            print(f"发现新的最佳模型! F1: {val_f1:.4f}")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"模型未改进。耐心计数: {patience_counter}/{patience}")

        # 检查早停条件
        if patience_counter >= patience:
            print(f"早停触发! 最佳验证F1: {best_f1:.4f}")
            break

    # 加载最佳模型
    if os.path.exists(best_model_path):
        print("加载最佳模型权重...")
        model.load_state_dict(torch.load(best_model_path))

    return model, training_stats

# 评估函数
def evaluate(model, eval_dataloader, device, id2tag, tag2id):
    model.eval()

    true_labels = []
    pred_labels = []

    # 用于seqeval评估的标签
    true_label_sequences = []
    pred_label_sequences = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # 将输入数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 检查是否有token_type_ids
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # 构建模型输入
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        # 只有在模型需要token_type_ids时才添加
        if token_type_ids is not None and 'token_type_ids' in model.forward.__code__.co_varnames:
            model_inputs['token_type_ids'] = token_type_ids

        # 不计算梯度
        with torch.no_grad():
            outputs = model(**model_inputs)

        # 获取预测结果
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

        # 遍历批次中的每个样本
        for i in range(input_ids.size(0)):
            sample_labels = []
            sample_preds = []

            # 遍历序列中的每个标记
            for j in range(input_ids.size(1)):
                if attention_mask[i, j].item() == 1 and labels[i, j].item() != -100:
                    # 只考虑有效标签
                    true_id = labels[i, j].item()
                    pred_id = predictions[i, j].item()

                    true_labels.append(true_id)
                    pred_labels.append(pred_id)

                    sample_labels.append(id2tag[true_id])
                    sample_preds.append(id2tag[pred_id])

            # 添加到序列列表
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

    # 返回报告、用于混淆矩阵的标记级标签，以及用于F1评分的实体级序列
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
    # 获取标签的唯一值，保留原始顺序
    unique_tags = sorted(list(set(true_tags + pred_tags)))

    # 创建混淆矩阵
    cm = np.zeros((len(unique_tags), len(unique_tags)), dtype=int)
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}

    for true_tag, pred_tag in zip(true_tags, pred_tags):
        cm[tag_to_idx[true_tag]][tag_to_idx[pred_tag]] += 1

    # 绘制热图
    plt.figure(figsize=(12, 10))

    # 检查是否过大的混淆矩阵
    if len(unique_tags) > 30:
        # 如果标签太多，绘制一个简化的热图
        sns.heatmap(cm, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (labels omitted due to size)')
    else:
        # 标签数量合理，绘制标准热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_tags, yticklabels=unique_tags)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

# 预测函数
def predict_sentence(model, tokenizer, sentence, tag2id, id2tag, device):
    model.eval()

    # Split the sentence into tokens
    tokens = sentence.split()

    # Tokenize the words
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors='pt'
    )

    # Move input data to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 检查是否有token_type_ids
    token_type_ids = encoding.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # 构建模型输入
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # 只有在模型需要token_type_ids时才添加
    if token_type_ids is not None and 'token_type_ids' in model.forward.__code__.co_varnames:
        model_inputs['token_type_ids'] = token_type_ids

    # No gradient calculation
    with torch.no_grad():
        outputs = model(**model_inputs)

    # Get prediction results
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    # Get word_ids mapping
    try:
        word_ids = encoding.word_ids()
        if word_ids is None:
            # If word_ids is None, we'll generate our own mapping
            # This is a workaround for when the tokenizer doesn't provide word_ids
            word_ids = []
            current_word_idx = None
            for token in tokenizer.convert_ids_to_tokens(input_ids[0]):
                if token.startswith("##") or token.startswith("Ġ") or token.startswith("▁"):
                    # This is a continuation of the previous word (for different tokenizers)
                    word_ids.append(current_word_idx)
                elif token in tokenizer.all_special_tokens:
                    # Special tokens don't map to original words
                    word_ids.append(None)
                else:
                    # This is a new word
                    if current_word_idx is None:
                        current_word_idx = 0
                    else:
                        current_word_idx += 1
                    word_ids.append(current_word_idx)
    except (AttributeError, ValueError) as e:
        print(f"Error getting word_ids: {e}")
        # Alternative approach: map tokens back to words manually
        word_ids = []
        sub_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        word_idx = -1
        for i, sub_token in enumerate(sub_tokens):
            if sub_token in tokenizer.all_special_tokens:
                word_ids.append(None)
            elif sub_token.startswith("##") or sub_token.startswith("Ġ") or sub_token.startswith("▁"):
                word_ids.append(word_idx)
            else:
                word_idx += 1
                word_ids.append(word_idx if word_idx < len(tokens) else None)

    # Store results
    results = []
    prev_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx < len(tokens):
            # Check if this is the first token of a word
            if word_idx != prev_word_idx:
                token = tokens[word_idx]
                label_id = predictions[0, i].item()
                label = id2tag[label_id]
                results.append((token, label))
            prev_word_idx = word_idx

    return results

def main():
    parser = argparse.ArgumentParser(description='Train Biomedical NER model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to BIO formatted data file')
    parser.add_argument('--output_dir', type=str, default='biomed_ner_model', help='Directory to save model and results')
    parser.add_argument('--pretrained_model', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                        help='Pretrained model to use (default: PubMedBERT)')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps for the scheduler')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据
    print("加载数据...")
    sentences, labels = load_data(args.data_path)
    print(f"加载了 {len(sentences)} 个句子")

    # 创建标签映射
    tag2id, id2tag = create_tag_mapping(labels)
    print(f"唯一标签: {list(tag2id.keys())}")

    # 将标签映射保存到文件
    with open(os.path.join(args.output_dir, "tag_mapping.json"), "w") as f:
        json.dump({"tag2id": tag2id, "id2tag": id2tag}, f)

    # 划分训练集和测试集
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=args.test_size, random_state=args.seed
    )

    print(f"训练集大小: {len(train_sentences)}")
    print(f"验证集大小: {len(val_sentences)}")

    # 加载预训练分词器
    print(f"加载 {args.pretrained_model} 分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # 创建数据集
    print("准备数据集...")
    train_dataset = NERDataset(train_sentences, train_labels, tokenizer, tag2id, max_len=args.max_len)
    val_dataset = NERDataset(val_sentences, val_labels, tokenizer, tag2id, max_len=args.max_len)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 加载预训练模型
    print(f"加载 {args.pretrained_model} 模型...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrained_model,
        num_labels=len(tag2id)
    )
    model.to(device)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 设置学习率调度器，添加预热步骤
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = min(args.warmup_steps, int(total_steps * 0.1))  # 默认为总步数的10%或用户指定值

    print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 训练模型
    print("开始训练模型...")
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
        patience=args.patience
    )

    # 可视化训练过程
    plot_training_stats(training_stats, args.output_dir)

    # 最终评估模型
    print("最终评估模型...")
    token_report, entity_report, true_tags, pred_tags, true_label_sequences, pred_label_sequences = evaluate(model, val_dataloader, device, id2tag, tag2id)

    print("标记级别分类报告:")
    print(token_report)

    print("\n实体级别分类报告:")
    print(entity_report)

    # 保存评估报告
    with open(os.path.join(args.output_dir, "evaluation_report.txt"), "w") as f:
        f.write("标记级别分类报告:\n")
        f.write(token_report)
        f.write("\n\n实体级别分类报告:\n")
        f.write(entity_report)

    # 可视化混淆矩阵
    plot_confusion_matrix(true_tags, pred_tags, args.output_dir)

    # 保存模型
    print("保存模型...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("模型训练完成!")

    # 示例预测
    print("\n示例预测:")
    test_sentences = [
        "effects of overexposure : may cause kidney and liver damage with prolonged exposure",
        "routes of entry : inhalation:yes skin:yes ingestion:yes health hazards acute and chronic",
        "can irritate the eyes and cause headache , nausea , vomiting",
        "ld50 lc50 mixture: oral ld50 (rat) 22,000 mg\/kg. low hazard to health. Reports of carcinogenicity: ntp:yes iarc:yes osha:no"
    ]

    for test_sentence in test_sentences:
        print(f"\n输入: {test_sentence}")
        results = predict_sentence(model, tokenizer, test_sentence, tag2id, id2tag, device)

        print("预测结果:")
        for token, label in results:
            print(f"{token}\t{label}")

if __name__ == "__main__":
    main()
    #python biobert-ner-train.py --data_path output0518_enhance.bio --pretrained_model allenai/scibert_scivocab_uncased --output_dir pubmedbert_ner_model0516 --batch_size 16 --epochs 10