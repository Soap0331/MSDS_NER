import os
import re
import json
from pathlib import Path

def parse_msds_file(file_path):
    """
    解析MSDS文件，提取产品ID、MSDS编号和危害识别文本
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取产品ID
    product_id_match = re.search(r'Product ID:([^\r\n]+)', content)
    product_id = product_id_match.group(1).strip() if product_id_match else ""

    # 提取MSDS编号
    msds_number_match = re.search(r'MSDS Number: ([^\r\n]+)', content)
    msds_number = msds_number_match.group(1).strip() if msds_number_match else ""

    # 提取危害识别部分
    hazards_text = ""

    # 查找危害识别部分的开始和结束标记
    start_marker = "=====================  Hazards Identification  ====================="
    end_marker = "=======================  First Aid Measures  ======================="

    start_index = content.find(start_marker)
    end_index = content.find(end_marker)

    if start_index != -1 and end_index != -1:
        # 找到标题行后的第一个换行符的位置
        title_end_pos = start_index + len(start_marker)
        content_start_pos = content.find('\n', title_end_pos) + 1

        # 提取危害识别内容
        hazards_raw_content = content[content_start_pos:end_index].strip()

        # 清洗文本：移除换行符和多余空格，转为小写
        # 1. 替换换行后跟空格的模式（通常是MSDS中的段落延续）
        cleaned_text = re.sub(r'\n\s+', ' ', hazards_raw_content)
        # 2. 替换任何剩余的换行符
        cleaned_text = re.sub(r'\n', ' ', cleaned_text)
        # 3. 替换多个空格为单个空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        # 4. 转为小写
        hazards_text = cleaned_text.lower()

    return {
        "document_id": Path(file_path).stem,
        "product_id": product_id,
        "msds_number": msds_number,
        "hazards_text": hazards_text,
        "raw_document": content
    }

def process_msds_directory(directory_path, output_file):
    """
    处理包含多个MSDS文件的目录，生成JSON数据集
    """
    dataset = []

    for file_path in Path(directory_path).glob("*.txt"):
        try:
            msds_data = parse_msds_file(file_path)
            dataset.append(msds_data)
            print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved to {output_file}")
    print(f"Total documents processed: {len(dataset)}")

def create_annotation_ready_format(input_json, output_json, format_type="basic"):
    """
    将处理好的数据转换为标注工具友好的格式
    format_type: "basic", "doccano", "label_studio"
    """
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if format_type == "doccano":
        # 使用JSONL格式（每行一个JSON对象）
        with open(output_json, 'w', encoding='utf-8') as f:
            for item in dataset:
                json_obj = {
                    "text": item["hazards_text"],
                    "meta": {
                        "product_id": item["product_id"],
                        "msds_number": item["msds_number"]
                    }
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        print(f"Doccano JSONL format dataset saved to {output_json}")
    elif format_type == "label_studio":
        # 创建Label Studio任务格式
        annotation_dataset = [
            {
                "data": {
                    "hazards_text": item["hazards_text"],
                    "product_id": item["product_id"],
                    "msds_number": item["msds_number"]
                }
            }
            for item in dataset
        ]
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(annotation_dataset, f, ensure_ascii=False, indent=2)
        print(f"Label Studio format dataset saved to {output_json}")
    else:
        # 基本格式，保持原样
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Basic format dataset saved to {output_json}")

# 使用示例
if __name__ == "__main__":
    # 1. 处理MSDS文件目录
    process_msds_directory("C:\\Users\\cubzz\\Desktop\\new_accident\\archive\\ingredients_name_cas_RTECS_wt_ext\\fsc_counter_useful\\f2_FSC\\FSC_6840", "./msds_dataset.json")

    # 2. 转换为标注工具友好的格式
    create_annotation_ready_format("msds_dataset.json", "msds_for_annotation.json", format_type="doccano")