import json
import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jKnowledgeGraphBuilder:
    def __init__(self, uri: str, username: str, password: str):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j数据库URI (例如: "bolt://localhost:7687")
            username: 数据库用户名
            password: 数据库密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

        # 定义实体类型到关系的映射
        self.entity_relations = {
            "Symptoms": ("Symptom", "HAS_SYMPTOM"),
            "TargetSite": ("TargetSite", "AFFECTS_ORGAN"),
            "Carcinogenicity": ("Carcinogenicity", "HAS_CARCINOGENICITY"),
            "NonCarcinogenicity": ("NonCarcinogenicity", "HAS_NON_CARCINOGENICITY"),
            "ToxicityDescription": ("ToxicityDescription", "HAS_TOXICITY"),
            "ToxicologicalDoseParameters": ("ToxicologicalDose", "HAS_DOSE_PARAMETER")
        }

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def read_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        读取JSONL文件

        Args:
            file_path: JSONL文件路径

        Returns:
            包含所有记录的列表
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            data.append(record)
                        except json.JSONDecodeError as e:
                            logger.error(f"第{line_num}行JSON解析错误: {e}")
                            continue
            logger.info(f"成功读取{len(data)}条记录")
            return data
        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return []
        except Exception as e:
            logger.error(f"读取文件时出错: {e}")
            return []

    def create_constraints_and_indexes(self):
        """创建约束和索引以优化性能"""
        constraints_and_indexes = [
            # 为Product节点创建唯一约束
            "CREATE CONSTRAINT product_msds_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.msds_number IS UNIQUE",

            # 为各种实体节点创建索引
            "CREATE INDEX symptom_text_index IF NOT EXISTS FOR (s:Symptom) ON (s.text)",
            "CREATE INDEX target_site_text_index IF NOT EXISTS FOR (t:TargetSite) ON (t.text)",
            "CREATE INDEX carcinogenicity_text_index IF NOT EXISTS FOR (c:Carcinogenicity) ON (c.text)",
            "CREATE INDEX non_carcinogenicity_text_index IF NOT EXISTS FOR (n:NonCarcinogenicity) ON (n.text)",
            "CREATE INDEX toxicity_text_index IF NOT EXISTS FOR (td:ToxicityDescription) ON (td.text)",
            "CREATE INDEX dose_text_index IF NOT EXISTS FOR (dose:ToxicologicalDose) ON (dose.text)"
        ]

        with self.driver.session() as session:
            for cypher in constraints_and_indexes:
                try:
                    session.run(cypher)
                    logger.info(f"执行成功: {cypher}")
                except Exception as e:
                    logger.warning(f"执行失败 {cypher}: {e}")

    def clear_database(self):
        """清空数据库（谨慎使用）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("数据库已清空")

    def create_product_node(self, session, product_id: str, msds_number: str):
        """创建产品节点"""
        cypher = """
        MERGE (p:Product {msds_number: $msds_number})
        SET p.product_id = $product_id
        RETURN p
        """
        session.run(cypher, msds_number=msds_number, product_id=product_id)

    def create_entity_and_relationship(self, session, msds_number: str, entity_text: str,
                                     entity_type: str, node_label: str, relationship_type: str):
        """创建实体节点和关系"""
        cypher = f"""
        MATCH (p:Product {{msds_number: $msds_number}})
        MERGE (e:{node_label} {{text: $entity_text}})
        MERGE (p)-[r:{relationship_type}]->(e)
        RETURN p, e, r
        """
        session.run(cypher, msds_number=msds_number, entity_text=entity_text)

    def process_record(self, session, record: Dict[str, Any]):
        """处理单条记录"""
        product_id = record.get('product_id', '')
        msds_number = record.get('msds_number', '')
        entities = record.get('entities', [])

        if not msds_number:
            logger.warning(f"记录缺少msds_number: {record}")
            return

        # 创建产品节点
        self.create_product_node(session, product_id, msds_number)

        # 处理实体
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            entity_type = entity.get('type', '')

            if not entity_text or entity_type not in self.entity_relations:
                continue

            node_label, relationship_type = self.entity_relations[entity_type]
            self.create_entity_and_relationship(
                session, msds_number, entity_text, entity_type, node_label, relationship_type
            )

    def build_knowledge_graph(self, jsonl_file_path: str, clear_db: bool = False):
        """
        构建知识图谱

        Args:
            jsonl_file_path: JSONL文件路径
            clear_db: 是否清空数据库
        """
        logger.info("开始构建知识图谱...")

        if clear_db:
            self.clear_database()

        # 创建约束和索引
        self.create_constraints_and_indexes()

        # 读取数据
        data = self.read_jsonl_file(jsonl_file_path)
        if not data:
            logger.error("没有数据可处理")
            return

        # 批量处理数据
        batch_size = 100
        processed_count = 0

        with self.driver.session() as session:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]

                # 使用事务处理批次
                with session.begin_transaction() as tx:
                    for record in batch:
                        try:
                            self.process_record(tx, record)
                            processed_count += 1
                        except Exception as e:
                            logger.error(f"处理记录时出错 {record.get('msds_number', 'unknown')}: {e}")

                logger.info(f"已处理 {processed_count}/{len(data)} 条记录")

        logger.info(f"知识图谱构建完成！总共处理了 {processed_count} 条记录")

    def get_graph_statistics(self):
        """获取图谱统计信息"""
        queries = {
            "产品数量": "MATCH (p:Product) RETURN count(p) as count",
            "症状数量": "MATCH (s:Symptom) RETURN count(s) as count",
            "靶器官数量": "MATCH (t:TargetSite) RETURN count(t) as count",
            "致癌性数量": "MATCH (c:Carcinogenicity) RETURN count(c) as count",
            "非致癌性数量": "MATCH (n:NonCarcinogenicity) RETURN count(n) as count",
            "毒性描述数量": "MATCH (td:ToxicityDescription) RETURN count(td) as count",
            "剂量参数数量": "MATCH (dose:ToxicologicalDose) RETURN count(dose) as count",
            "总关系数量": "MATCH ()-[r]->() RETURN count(r) as count"
        }

        stats = {}
        with self.driver.session() as session:
            for name, query in queries.items():
                result = session.run(query)
                stats[name] = result.single()['count']

        return stats

# 数据验证和分析工具
class DataAnalyzer:
    @staticmethod
    def analyze_jsonl_file(file_path: str):
        """分析JSONL文件内容"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data.append(json.loads(line))

        # 统计信息
        total_records = len(data)
        entity_type_counts = {}
        msds_numbers = set()

        for record in data:
            msds_numbers.add(record.get('msds_number', ''))
            for entity in record.get('entities', []):
                entity_type = entity.get('type', '')
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        print(f"数据文件分析结果:")
        print(f"- 总记录数: {total_records}")
        print(f"- 唯一MSDS编号数: {len(msds_numbers)}")
        print(f"- 实体类型统计:")
        for entity_type, count in sorted(entity_type_counts.items()):
            print(f"  - {entity_type}: {count}")

# 使用示例
def main():
    # 配置Neo4j连接信息
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "20000331"

    # JSONL文件路径
    JSONL_FILE_PATH = "output_entities_meta.jsonl"

    try:
        # 1. 分析数据文件
        print("=== 数据文件分析 ===")
        DataAnalyzer.analyze_jsonl_file(JSONL_FILE_PATH)
        print()

        # 2. 构建知识图谱
        print("=== 构建知识图谱 ===")
        builder = Neo4jKnowledgeGraphBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

        # 构建图谱（clear_db=True 会清空现有数据）
        builder.build_knowledge_graph(JSONL_FILE_PATH, clear_db=True)

        # 3. 查看统计信息
        print("\n=== 图谱统计信息 ===")
        stats = builder.get_graph_statistics()
        for name, count in stats.items():
            print(f"{name}: {count}")

        builder.close()
        print("\n知识图谱构建完成！")

    except Exception as e:
        logger.error(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()