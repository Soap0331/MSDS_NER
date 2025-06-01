import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jIngredientExtender:
    def __init__(self, uri: str, username: str, password: str):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j数据库URI (例如: "bolt://localhost:7687")
            username: 数据库用户名
            password: 数据库密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def read_ingredient_csv(self, file_path: str) -> pd.DataFrame:
        """
        读取成分CSV文件

        Args:
            file_path: CSV文件路径

        Returns:
            包含成分数据的DataFrame
        """
        try:
            df = pd.read_csv(file_path)

            # 验证必需的列是否存在
            required_columns = ['product_id', 'msds_number', 'ingredient_name', 'cas', 'fraction']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"CSV文件缺少必需的列: {missing_columns}")

            # 数据清洗
            df = df.dropna(subset=['msds_number', 'ingredient_name'])  # 删除关键字段为空的行
            df['cas'] = df['cas'].fillna('')  # CAS号为空时填充空字符串

            # 将fraction转换为字符串，保持原始格式
            df['fraction'] = df['fraction'].astype(str)
            # 处理NaN值，将其转换为空字符串
            df['fraction'] = df['fraction'].replace('nan', '')
            # 去除首尾空格
            df['fraction'] = df['fraction'].str.strip()
            # 如果fraction为空，设为空字符串
            df['fraction'] = df['fraction'].fillna('')

            logger.info(f"成功读取CSV文件，共{len(df)}条记录")
            return df

        except Exception as e:
            logger.error(f"读取CSV文件时出错: {e}")
            raise

    def create_ingredient_constraints_and_indexes(self):
        """创建成分相关的约束和索引"""
        constraints_and_indexes = [
            # 为Ingredient节点创建唯一约束
            "CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",

            # 为CAS节点创建唯一约束
            "CREATE CONSTRAINT cas_number_unique IF NOT EXISTS FOR (c:CAS) REQUIRE c.cas_number IS UNIQUE",

            # 创建索引以提升查询性能
            "CREATE INDEX ingredient_name_index IF NOT EXISTS FOR (i:Ingredient) ON (i.name)",
            "CREATE INDEX cas_number_index IF NOT EXISTS FOR (c:CAS) ON (c.cas_number)"
        ]

        with self.driver.session() as session:
            for cypher in constraints_and_indexes:
                try:
                    session.run(cypher)
                    logger.info(f"执行成功: {cypher}")
                except Exception as e:
                    logger.warning(f"执行失败 {cypher}: {e}")

    def create_ingredient_and_cas_nodes(self, session, ingredient_name: str, cas_number: str):
        """
        创建成分节点和CAS节点，并建立它们之间的关系

        Args:
            session: Neo4j会话
            ingredient_name: 成分名称
            cas_number: CAS号
        """
        # 创建成分节点
        ingredient_cypher = """
        MERGE (i:Ingredient {name: $ingredient_name})
        RETURN i
        """
        session.run(ingredient_cypher, ingredient_name=ingredient_name)

        # 如果有CAS号，创建CAS节点并建立关系
        if cas_number and cas_number.strip():
            cas_cypher = """
            MERGE (c:CAS {cas_number: $cas_number})
            WITH c
            MATCH (i:Ingredient {name: $ingredient_name})
            MERGE (i)-[:HAS_CAS]->(c)
            RETURN i, c
            """
            session.run(cas_cypher, cas_number=cas_number.strip(), ingredient_name=ingredient_name)

    def create_product_ingredient_relationship(self, session, msds_number: str,
                                             ingredient_name: str, fraction: str):
        """
        创建产品到成分的关系（包含比例信息，以字符串形式存储）

        Args:
            session: Neo4j会话
            msds_number: MSDS编号
            ingredient_name: 成分名称
            fraction: 成分比例（字符串形式）
        """
        cypher = """
        MATCH (p:Product {msds_number: $msds_number})
        MATCH (i:Ingredient {name: $ingredient_name})
        MERGE (p)-[r:HAS_INGREDIENT]->(i)
        SET r.fraction = $fraction
        RETURN p, r, i
        """
        session.run(cypher, msds_number=msds_number, ingredient_name=ingredient_name, fraction=fraction)

    def process_ingredient_batch(self, session, batch_df: pd.DataFrame):
        """
        批量处理成分数据

        Args:
            session: Neo4j会话
            batch_df: 成分数据批次
        """
        for _, row in batch_df.iterrows():
            msds_number = row['msds_number']
            ingredient_name = row['ingredient_name']
            cas_number = row['cas']
            fraction = row['fraction']  # 现在是字符串类型

            try:
                # 1. 创建成分和CAS节点
                self.create_ingredient_and_cas_nodes(session, ingredient_name, cas_number)

                # 2. 创建产品到成分的关系（fraction以字符串形式存储）
                self.create_product_ingredient_relationship(session, msds_number, ingredient_name, fraction)

            except Exception as e:
                logger.error(f"处理成分记录时出错 {msds_number}-{ingredient_name}: {e}")

    def calculate_shared_ingredients(self):
        """
        计算产品之间的共享成分关系
        """
        logger.info("开始计算产品共享成分关系...")

        # 查询每个产品的成分
        query = """
        MATCH (p:Product)-[:HAS_INGREDIENT]->(i:Ingredient)
        RETURN p.msds_number as msds_number, collect(i.name) as ingredients
        """

        product_ingredients = {}
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                product_ingredients[record['msds_number']] = set(record['ingredients'])

        # 计算产品之间的共享成分
        shared_relationships = []
        product_list = list(product_ingredients.keys())

        for i, product_a in enumerate(product_list):
            for product_b in product_list[i+1:]:
                ingredients_a = product_ingredients[product_a]
                ingredients_b = product_ingredients[product_b]

                # 计算共同成分数量
                shared_count = len(ingredients_a.intersection(ingredients_b))

                # 如果有共同成分，记录关系
                if shared_count > 0:
                    shared_relationships.append((product_a, product_b, shared_count))

        logger.info(f"发现 {len(shared_relationships)} 对产品存在共享成分")
        return shared_relationships

    def create_shared_ingredient_relationships(self, shared_relationships: List[Tuple[str, str, int]]):
        """
        创建产品间的共享成分关系

        Args:
            shared_relationships: 共享成分关系列表 [(product_a, product_b, count), ...]
        """
        logger.info("创建产品共享成分关系...")

        cypher = """
        MATCH (p1:Product {msds_number: $msds_a})
        MATCH (p2:Product {msds_number: $msds_b})
        MERGE (p1)-[r:SHARED_INGREDIENT]-(p2)
        SET r.count = $count
        RETURN p1, r, p2
        """

        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                for msds_a, msds_b, count in shared_relationships:
                    try:
                        tx.run(cypher, msds_a=msds_a, msds_b=msds_b, count=count)
                    except Exception as e:
                        logger.error(f"创建共享关系时出错 {msds_a}-{msds_b}: {e}")

    def extend_knowledge_graph(self, csv_file_path: str, batch_size: int = 100):
        """
        扩展知识图谱，添加成分信息

        Args:
            csv_file_path: CSV文件路径
            batch_size: 批处理大小
        """
        logger.info("开始扩展知识图谱，添加成分信息...")

        # 1. 创建约束和索引
        self.create_ingredient_constraints_and_indexes()

        # 2. 读取CSV数据
        df = self.read_ingredient_csv(csv_file_path)

        if df.empty:
            logger.error("没有数据可处理")
            return

        # 3. 批量处理成分数据
        processed_count = 0
        total_count = len(df)

        with self.driver.session() as session:
            for i in range(0, total_count, batch_size):
                batch_df = df.iloc[i:i+batch_size]

                # 使用事务处理批次
                with session.begin_transaction() as tx:
                    self.process_ingredient_batch(tx, batch_df)
                    processed_count += len(batch_df)

                logger.info(f"已处理 {processed_count}/{total_count} 条成分记录")

        # 4. 计算和创建产品间的共享成分关系
        shared_relationships = self.calculate_shared_ingredients()
        self.create_shared_ingredient_relationships(shared_relationships)

        logger.info(f"知识图谱扩展完成！处理了 {processed_count} 条成分记录，创建了 {len(shared_relationships)} 个共享关系")

    def get_ingredient_statistics(self):
        """获取成分相关的统计信息"""
        queries = {
            "成分数量": "MATCH (i:Ingredient) RETURN count(i) as count",
            "CAS号数量": "MATCH (c:CAS) RETURN count(c) as count",
            "产品-成分关系数量": "MATCH ()-[r:HAS_INGREDIENT]->() RETURN count(r) as count",
            "成分-CAS关系数量": "MATCH ()-[r:HAS_CAS]->() RETURN count(r) as count",
            "产品共享关系数量": "MATCH ()-[r:SHARED_INGREDIENT]-() RETURN count(r)/2 as count",
            "有成分信息的产品数量": """
                MATCH (p:Product)-[:HAS_INGREDIENT]->(:Ingredient)
                RETURN count(DISTINCT p) as count
            """,
            "平均每个产品的成分数量": """
                MATCH (p:Product)-[:HAS_INGREDIENT]->(:Ingredient)
                WITH p, count(*) as ingredient_count
                RETURN avg(ingredient_count) as count
            """
        }

        stats = {}
        with self.driver.session() as session:
            for name, query in queries.items():
                try:
                    result = session.run(query)
                    record = result.single()
                    stats[name] = round(record['count'], 2) if record and record['count'] else 0
                except Exception as e:
                    logger.error(f"执行统计查询时出错 {name}: {e}")
                    stats[name] = 0

        return stats

    def get_sample_queries(self):
        """提供一些示例查询来验证数据"""
        sample_queries = {
            "查看某产品的所有成分": """
                MATCH (p:Product {msds_number: 'YOUR_MSDS_NUMBER'})-[r:HAS_INGREDIENT]->(i:Ingredient)
                OPTIONAL MATCH (i)-[:HAS_CAS]->(c:CAS)
                RETURN p.msds_number, i.name, r.fraction, c.cas_number
                ORDER BY i.name
            """,

            "查找共享成分最多的产品对": """
                MATCH (p1:Product)-[r:SHARED_INGREDIENT]-(p2:Product)
                WHERE p1.msds_number < p2.msds_number
                RETURN p1.msds_number, p2.msds_number, r.count
                ORDER BY r.count DESC
                LIMIT 10
            """,

            "查找特定成分的所有产品": """
                MATCH (i:Ingredient {name: 'YOUR_INGREDIENT_NAME'})<-[r:HAS_INGREDIENT]-(p:Product)
                RETURN p.msds_number, r.fraction
                ORDER BY p.msds_number
            """,

            "查找特定CAS号的相关信息": """
                MATCH (c:CAS {cas_number: 'YOUR_CAS_NUMBER'})<-[:HAS_CAS]-(i:Ingredient)<-[r:HAS_INGREDIENT]-(p:Product)
                RETURN c.cas_number, i.name, p.msds_number, r.fraction
                ORDER BY p.msds_number
            """,

            "查看不同fraction值的分布": """
                MATCH ()-[r:HAS_INGREDIENT]->()
                RETURN r.fraction, count(*) as count
                ORDER BY count DESC
                LIMIT 20
            """,

            "查找fraction为特定值的关系": """
                MATCH (p:Product)-[r:HAS_INGREDIENT]->(i:Ingredient)
                WHERE r.fraction = 'YOUR_FRACTION_VALUE'
                RETURN p.msds_number, i.name, r.fraction
            """
        }

        return sample_queries

# 数据验证工具
class IngredientDataAnalyzer:
    @staticmethod
    def analyze_csv_file(file_path: str):
        """分析CSV文件内容"""
        try:
            df = pd.read_csv(file_path)

            print(f"CSV文件分析结果:")
            print(f"- 总记录数: {len(df)}")
            print(f"- 列名: {list(df.columns)}")
            print(f"- 唯一MSDS编号数: {df['msds_number'].nunique()}")
            print(f"- 唯一成分数: {df['ingredient_name'].nunique()}")
            print(f"- 唯一CAS号数: {df['cas'].nunique()}")

            # 分析fraction字段（作为字符串）
            fraction_str = df['fraction'].astype(str)
            print(f"- Fraction字段统计（字符串形式）:")
            print(f"  - 唯一值数量: {fraction_str.nunique()}")
            print(f"  - 最常见的值:")
            for value, count in fraction_str.value_counts().head(10).items():
                print(f"    '{value}': {count}次")

            # 检查数据质量
            print(f"- 数据质量检查:")
            print(f"  - MSDS编号缺失: {df['msds_number'].isna().sum()}")
            print(f"  - 成分名称缺失: {df['ingredient_name'].isna().sum()}")
            print(f"  - CAS号缺失: {df['cas'].isna().sum()}")
            print(f"  - 比例缺失: {df['fraction'].isna().sum()}")

        except Exception as e:
            print(f"分析CSV文件时出错: {e}")

# 使用示例
def main():
    # 配置Neo4j连接信息
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "20000331"

    # CSV文件路径
    CSV_FILE_PATH = "msds_6840_ingredients_with_MSDSnumber.csv"  # 请替换为你的CSV文件路径

    try:
        # 1. 分析CSV文件
        print("=== CSV文件分析 ===")
        IngredientDataAnalyzer.analyze_csv_file(CSV_FILE_PATH)
        print()

        # 2. 扩展知识图谱
        print("=== 扩展知识图谱 ===")
        extender = Neo4jIngredientExtender(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

        # 执行扩展
        extender.extend_knowledge_graph(CSV_FILE_PATH)

        # 3. 查看统计信息
        print("\n=== 成分信息统计 ===")
        stats = extender.get_ingredient_statistics()
        for name, count in stats.items():
            print(f"{name}: {count}")

        # 4. 提供示例查询
        print("\n=== 示例查询 ===")
        sample_queries = extender.get_sample_queries()
        for name, query in sample_queries.items():
            print(f"\n{name}:")
            print(query)

        extender.close()
        print("\n知识图谱扩展完成！")

    except Exception as e:
        logger.error(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()