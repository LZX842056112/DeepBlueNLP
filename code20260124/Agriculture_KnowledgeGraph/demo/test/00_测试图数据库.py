# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/8/30 14:49
Create User : 19410
Desc : xxx
"""

from py2neo import Graph
from py2neo.cypher import Cursor

graph = Graph("bolt://127.0.0.1:7687", name="agriculture", auth=("neo4j", "Lzx19960802"))

database = "agriculture"  # 对应的数据库

print("=" * 100)
# r2 = graph.run("""MATCH (n) RETURN n limit 1""")
r2 = graph.run(""" MATCH (entity1:NewNode)-[rel]->(entity2)  RETURN rel,entity2 limit 1; """)
print(r2.data())
