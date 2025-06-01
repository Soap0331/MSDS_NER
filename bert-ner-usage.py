import argparse
import os
import json
import torch
from transformers import BertTokenizer, BertTokenizerFast, BertForTokenClassification

def load_model(model_dir):
    """
    从指定目录加载预训练的BERT NER模型
    """
    # 加载标签映射
    with open(os.path.join(model_dir, "tag_mapping.json"), "r") as f:
        mapping = json.load(f)
        tag2id = mapping["tag2id"]
        id2tag = {int(k): v for k, v in mapping["id2tag"].items()}

    # 尝试加载Fast版本的分词器，如果失败则回退到标准版本
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        print("Using fast tokenizer")
    except:
        print("Fast tokenizer not available, falling back to standard tokenizer")
        tokenizer = BertTokenizer.from_pretrained(model_dir)

    # 加载模型
    model = BertForTokenClassification.from_pretrained(model_dir)

    return model, tokenizer, tag2id, id2tag

def predict_sentence(model, tokenizer, sentence, tag2id, id2tag, device):
    """
    对单个句子进行命名实体识别
    返回每个单词和对应的预测标签
    使用与BIOlabel.py相似的分词逻辑
    """
    model.eval()
    model.to(device)

    # 使用与BIOlabel.py一致的分词方法
    # 创建一个与正则表达式匹配的词列表
    import re
    tokens = []
    token_spans = []
    for match in re.finditer(r'\b\w+\b|[^\w\s]', sentence):
        start, end = match.span()
        word = match.group()
        tokens.append(word)
        token_spans.append((start, end))

    # 如果没有内容，返回空结果
    if not tokens:
        return []

    # 对单词进行BERT分词
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512  # 根据需要调整长度
    )

    # 将输入数据移动到设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # token_type_ids 可能不存在于某些分词器中
    token_type_ids = encoding.get('token_type_ids')

    # 准备模型输入
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    # token_type_ids 只有在 BERT 模型中才需要
    if token_type_ids is not None:
        model_inputs['token_type_ids'] = token_type_ids.to(device)

    # 不计算梯度
    with torch.no_grad():
        outputs = model(**model_inputs)

    # 获取预测结果
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    # 尝试获取 word_ids 映射
    try:
        word_ids = encoding.word_ids()
        if word_ids is None:
            raise AttributeError("word_ids() returned None")
    except (AttributeError, ValueError) as e:
        print(f"Using manual token mapping due to: {e}")
        # 如果 word_ids() 不可用，手动创建映射
        word_ids = []
        sub_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        word_idx = -1
        for i, sub_token in enumerate(sub_tokens):
            if sub_token in tokenizer.all_special_tokens:
                word_ids.append(None)
            elif sub_token.startswith("##"):
                word_ids.append(word_idx)
            else:
                word_idx += 1
                word_ids.append(word_idx if word_idx < len(tokens) else None)

    # 存储结果
    results = []
    prev_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx < len(tokens):
            # 检查是否为单词的第一个标记
            if word_idx != prev_word_idx:
                token = tokens[word_idx]
                label_id = predictions[0, i].item()
                label = id2tag[label_id]
                results.append((token, label))
            prev_word_idx = word_idx

    return results

def extract_entities(results):
    """
    从BIO标签结果中提取实体
    返回一个列表，每个元素是(实体文本, 实体类型)

    基于BIOlabel.py中的逻辑重新实现，确保与训练数据的BIO标签生成方式一致
    """
    entities = []
    current_entity = []
    current_type = None

    for i, (token, tag) in enumerate(results):
        if tag.startswith('B-'):
            # 如果我们有一个正在处理的实体，先保存它
            if current_entity:
                entities.append((' '.join(current_entity), current_type.split('-', 1)[1]))
                current_entity = []

            # 开始一个新实体
            current_entity = [token]
            current_type = tag

        elif tag.startswith('I-'):
            # 检查I标签类型是否与当前实体类型匹配
            if current_entity and current_type and tag[2:] == current_type[2:]:
                # 继续当前实体
                current_entity.append(token)
            elif not current_entity:
                # 如果没有前导B标签，但按照BIOlabel.py的处理方式，
                # 我们将I标签视为B标签开始一个新实体
                current_entity = [token]
                current_type = "B-" + tag[2:]  # 将I-转换为B-
            else:
                # I标签类型与当前实体类型不匹配
                # 保存当前实体并开始新实体
                entities.append((' '.join(current_entity), current_type.split('-', 1)[1]))
                current_entity = [token]
                current_type = "B-" + tag[2:]  # 将I-转换为B-

        else:  # O标签
            # 如果我们有一个正在处理的实体，保存它
            if current_entity:
                entities.append((' '.join(current_entity), current_type.split('-', 1)[1]))
                current_entity = []
                current_type = None

    # 检查最后一个实体
    if current_entity:
        entities.append((' '.join(current_entity), current_type.split('-', 1)[1]))

    return entities

def predict_from_file(model, tokenizer, input_file, output_file, tag2id, id2tag, device):
    """
    从文件读取句子进行预测，并将结果写入输出文件
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(f"Sentence: {sentence}\n")

            # 预测
            results = predict_sentence(model, tokenizer, sentence, tag2id, id2tag, device)

            # 写入标记级别的结果
            f.write("\nToken-level predictions:\n")
            for token, label in results:
                f.write(f"{token}\t{label}\n")

            # 提取实体
            entities = extract_entities(results)

            # 写入实体级别的结果
            f.write("\nExtracted entities:\n")
            if entities:
                for entity_text, entity_type in entities:
                    f.write(f"{entity_type}: {entity_text}\n")
            else:
                f.write("No entities found.\n")

            f.write("\n" + "="*50 + "\n\n")

def process_jsonl_file(model, tokenizer, input_file, output_jsonl, output_bio, tag2id, id2tag, device):
    """
    处理JSONL文件，提取实体，并将结果写入输出文件
    使用与BIOlabel.py相似的处理逻辑

    Args:
        model: BERT NER模型
        tokenizer: BERT分词器
        input_file: 输入JSONL文件路径
        output_jsonl: 输出JSONL文件路径
        output_bio: 输出BIO文件路径
        tag2id: 标签到ID的映射
        id2tag: ID到标签的映射
        device: 计算设备 (cuda或cpu)
    """
    # 读取并处理JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    results = []
    bio_annotations = []

    for item in data:
        text = item.get('text', '')
        meta = item.get('meta', {})

        if not text:
            continue

        # 进行令牌级别的预测
        token_predictions = predict_sentence(model, tokenizer, text, tag2id, id2tag, device)

        # 提取实体
        entities = extract_entities(token_predictions)

        # 按照BIOlabel.py中的方式处理实体
        # 将实体转换为字符级别的标签
        char_level_entities = []
        current_position = 0

        # 将token位置映射到字符位置
        import re
        token_positions = []
        for match in re.finditer(r'\b\w+\b|[^\w\s]', text):
            start, end = match.span()
            token_positions.append((start, end, match.group()))

        # 生成字符级实体
        for entity_text, entity_type in entities:
            entity_tokens = entity_text.split()

            # 查找这些token在原文中的位置
            for i, (start, end, token) in enumerate(token_positions):
                tokens_to_match = token_positions[i:i+len(entity_tokens)]
                if len(tokens_to_match) < len(entity_tokens):
                    continue

                potential_match = " ".join([t[2] for t in tokens_to_match[:len(entity_tokens)]])
                if potential_match.lower() == entity_text.lower():
                    # 找到匹配，添加字符级实体
                    entity_start = tokens_to_match[0][0]
                    entity_end = tokens_to_match[len(entity_tokens)-1][1]
                    char_level_entities.append({
                        'text': text[entity_start:entity_end],
                        'start': entity_start,
                        'end': entity_end,
                        'type': entity_type
                    })
                    break

        # 创建JSONL输出的结果对象
        result = {
            'text': text,
            'meta': meta,
            'entities': [
                {'text': entity['text'], 'type': entity['type']}
                for entity in char_level_entities
            ]
        }
        results.append(result)

        # 添加BIO注释 - 使用与BIOlabel.py相似的格式
        bio_entry = {
            'text': text,
            'tokens': [token for token, _ in token_predictions],
            'tags': [tag for _, tag in token_predictions],
            'meta': meta
        }
        bio_annotations.append(bio_entry)

    # 写入输出JSONL文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 写入输出BIO文件 - 模仿BIOlabel.py的输出格式
    with open(output_bio, 'w', encoding='utf-8') as f:
        for bio_entry in bio_annotations:
            # 将元数据写为注释
            f.write(f"# meta: {json.dumps(bio_entry['meta'], ensure_ascii=False)}\n")

            # 写入令牌和标签
            for token, tag in zip(bio_entry['tokens'], bio_entry['tags']):
                f.write(f"{token}\t{tag}\n")

            # 句子之间的空行
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Use BERT NER model for prediction')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with saved model')
    parser.add_argument('--mode', type=str, choices=['interactive', 'file', 'jsonl'], default='interactive',
                       help='Prediction mode: interactive, from file, or from JSONL')
    parser.add_argument('--input_file', type=str, help='Input file with sentences or JSONL data')
    parser.add_argument('--output_file', type=str, help='Output file for predictions (for file mode)')
    parser.add_argument('--output_jsonl', type=str, help='Output JSONL file for entity results (for jsonl mode)')
    parser.add_argument('--output_bio', type=str, help='Output BIO file for annotations (for jsonl mode)')

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading model from {args.model_dir}...")
    model, tokenizer, tag2id, id2tag = load_model(args.model_dir)
    model.to(device)

    if args.mode == 'interactive':
        print("\nInteractive mode. Enter a sentence for NER prediction. Type 'quit' to exit.")

        while True:
            sentence = input("\nEnter sentence: ")

            if sentence.lower() == 'quit':
                break

            if not sentence:
                continue

            # 预测
            results = predict_sentence(model, tokenizer, sentence, tag2id, id2tag, device)

            # 打印标记级别的结果
            print("\nToken-level predictions:")
            for token, label in results:
                print(f"{token}\t{label}")

            # 提取实体
            entities = extract_entities(results)

            # 打印实体级别的结果
            print("\nExtracted entities:")
            if entities:
                for entity_text, entity_type in entities:
                    print(f"{entity_type}: {entity_text}")
            else:
                print("No entities found.")

    elif args.mode == 'file':
        if not args.input_file or not args.output_file:
            print("Error: input_file and output_file are required for file mode")
            return

        print(f"Processing sentences from {args.input_file}...")
        predict_from_file(model, tokenizer, args.input_file, args.output_file, tag2id, id2tag, device)
        print(f"Results written to {args.output_file}")

    else:  # jsonl mode
        if not args.input_file or not args.output_jsonl or not args.output_bio:
            print("Error: input_file, output_jsonl, and output_bio are required for jsonl mode")
            return

        print(f"Processing JSONL data from {args.input_file}...")
        process_jsonl_file(model, tokenizer, args.input_file, args.output_jsonl, args.output_bio, tag2id, id2tag, device)
        print(f"Results written to {args.output_jsonl} and {args.output_bio}")

if __name__ == "__main__":
    main()
    #最终使用训练结束模型提取实体的文件
    #python bert-ner-usage.py --mode jsonl --model_dir ner_model_new --input_file msds_for_annotation.jsonl --output_jsonl ./extract_result/output_results.jsonl --output_bio ./extract_result/output_annotations.bio