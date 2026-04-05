#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将 GraphRAG 输出导入到 Neo4j 知识图谱
"""

import pandas as pd
import os
from pathlib import Path
from neo4j import GraphDatabase
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="graphrag-neo4j-import")

class GraphRAGToNeo4j:
    def __init__(self, graphrag_output_dir: str):
        """
        初始化导入器
        
        Args:
            graphrag_output_dir: GraphRAG 输出目录路径
        """
        self.output_dir = Path(graphrag_output_dir)
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        logger.info(f"连接到 Neo4j: {settings.NEO4J_URL}")
        
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        
    def clear_database(self):
        """清空数据库（可选）"""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            logger.info("清空现有数据...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("数据库已清空")
    
    def create_indexes(self):
        """创建索引以提高性能"""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            logger.info("创建索引...")
            
            # 为实体创建索引 - 使用 title 字段
            session.run("""
                CREATE INDEX entity_id IF NOT EXISTS 
                FOR (e:Entity) ON (e.id)
            """)
            
            session.run("""
                CREATE INDEX entity_title IF NOT EXISTS 
                FOR (e:Entity) ON (e.title)
            """)
            
            logger.info("索引创建完成")
    
    def import_entities(self):
        """导入实体节点"""
        entities_file = self.output_dir / "entities.parquet"
        
        if not entities_file.exists():
            logger.error(f"找不到实体文件: {entities_file}")
            return
        
        logger.info(f"读取实体数据: {entities_file}")
        df = pd.read_parquet(entities_file)
        
        logger.info(f"共有 {len(df)} 个实体")
        logger.info(f"实体列: {df.columns.tolist()}")
        
        # 批量导入实体
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                entities = []
                for _, row in batch.iterrows():
                    entity = {
                        'id': str(row.get('id', '')),
                        'title': str(row.get('title', '')),  # 使用 title 而不是 name
                        'type': str(row.get('type', 'Unknown')),
                        'description': str(row.get('description', '')),
                        'human_readable_id': str(row.get('human_readable_id', '')),
                        'text_unit_ids': str(row.get('text_unit_ids', ''))
                    }
                    entities.append(entity)
                
                # 批量创建节点
                session.run("""
                    UNWIND $entities AS entity
                    MERGE (e:Entity {id: entity.id})
                    SET e.title = entity.title,
                        e.type = entity.type,
                        e.description = entity.description,
                        e.human_readable_id = entity.human_readable_id,
                        e.text_unit_ids = entity.text_unit_ids
                """, entities=entities)
                
                logger.info(f"已导入实体批次 {batch_num}/{total_batches}")
        
        logger.info("实体导入完成")
    
    def import_relationships(self):
        """导入关系边"""
        relationships_file = self.output_dir / "relationships.parquet"
        
        if not relationships_file.exists():
            logger.error(f"找不到关系文件: {relationships_file}")
            return
        
        logger.info(f"读取关系数据: {relationships_file}")
        df = pd.read_parquet(relationships_file)
        
        logger.info(f"共有 {len(df)} 个关系")
        logger.info(f"关系列: {df.columns.tolist()}")
        
        # 批量导入关系
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                relationships = []
                for _, row in batch.iterrows():
                    rel = {
                        'source': str(row.get('source', '')),
                        'target': str(row.get('target', '')),
                        'description': str(row.get('description', '')),
                        'weight': float(row.get('weight', 1.0)) if pd.notna(row.get('weight')) else 1.0,
                        'text_unit_ids': str(row.get('text_unit_ids', ''))
                    }
                    relationships.append(rel)
                
                # 批量创建关系 - 使用 title 字段匹配
                session.run("""
                    UNWIND $relationships AS rel
                    MATCH (source:Entity {title: rel.source})
                    MATCH (target:Entity {title: rel.target})
                    MERGE (source)-[r:RELATED_TO]->(target)
                    SET r.description = rel.description,
                        r.weight = rel.weight,
                        r.text_unit_ids = rel.text_unit_ids
                """, relationships=relationships)
                
                logger.info(f"已导入关系批次 {batch_num}/{total_batches}")
        
        logger.info("关系导入完成")
    
    def import_communities(self):
        """导入社区信息（可选）"""
        communities_file = self.output_dir / "communities.parquet"
        
        if not communities_file.exists():
            logger.warning(f"找不到社区文件: {communities_file}")
            return
        
        logger.info(f"读取社区数据: {communities_file}")
        df = pd.read_parquet(communities_file)
        
        logger.info(f"共有 {len(df)} 个社区")
        
        # 批量导入社区
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                communities = []
                for _, row in batch.iterrows():
                    community = {
                        'id': str(row.get('id', '')),
                        'title': str(row.get('title', '')),
                        'level': int(row.get('level', 0)) if pd.notna(row.get('level')) else 0,
                        'entity_ids': str(row.get('entity_ids', ''))
                    }
                    communities.append(community)
                
                # 创建社区节点
                session.run("""
                    UNWIND $communities AS community
                    MERGE (c:Community {id: community.id})
                    SET c.title = community.title,
                        c.level = community.level,
                        c.entity_ids = community.entity_ids
                """, communities=communities)
                
                logger.info(f"已导入社区批次 {batch_num}/{total_batches}")
        
        logger.info("社区导入完成")
    
    def get_statistics(self):
        """获取导入统计信息"""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            # 统计节点数
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()['count']
            
            # 统计关系数
            result = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            
            # 统计社区数
            result = session.run("MATCH (c:Community) RETURN count(c) as count")
            community_count = result.single()['count']
            
            logger.info("=" * 50)
            logger.info("导入统计:")
            logger.info(f"  实体节点数: {entity_count}")
            logger.info(f"  关系边数: {rel_count}")
            logger.info(f"  社区数: {community_count}")
            logger.info("=" * 50)

def main():
    """主函数"""
    # GraphRAG 输出目录 - 直接使用 output 目录
    graphrag_output_dir = r"C:\Users\L\Desktop\智能客服Agent\Agent\code\code\backend\deepseek_agent\llm_backend\app\graphrag\data\output"
    
    # 检查目录是否存在
    if not os.path.exists(graphrag_output_dir):
        logger.error(f"输出目录不存在: {graphrag_output_dir}")
        logger.info("请检查 GraphRAG 输出目录路径")
        return
    
    try:
        # 创建导入器
        importer = GraphRAGToNeo4j(graphrag_output_dir)
        
        # 询问是否清空数据库
        print("\n是否清空现有 Neo4j 数据库? (y/n): ", end="")
        choice = input().strip().lower()
        if choice == 'y':
            importer.clear_database()
        
        # 创建索引
        importer.create_indexes()
        
        # 导入数据
        logger.info("开始导入数据...")
        importer.import_entities()
        importer.import_relationships()
        importer.import_communities()
        
        # 显示统计信息
        importer.get_statistics()
        
        # 关闭连接
        importer.close()
        
        logger.info("导入完成！")
        logger.info(f"可以在 Neo4j Browser 中查看: {settings.NEO4J_URL.replace('bolt://', 'http://')}")
        
    except Exception as e:
        logger.error(f"导入过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

