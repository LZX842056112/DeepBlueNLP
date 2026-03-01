# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/29 20:51
Create User : 19410
Desc : xxx
"""
# 导入强大的数据分析处理库 Pandas，常用于读取本地结构化文件
import pandas as pd

if __name__ == '__main__':
    # 再次使用 py2neo 库操作数据库
    from py2neo import Graph
    from py2neo.cypher import Cursor

    # 第一步：配置并连接数据库
    # 使用 neo4j:// 协议连接本地数据库（支持路由，比 bolt:// 更推荐）
    # profile = "bolt://127.0.0.1:7687"  # 连接图数据库的url
    profile = "neo4j://127.0.0.1:7687"  # 连接图数据库的url

    # 连接到名为 'nlp' 的数据库 (之前是 'neo4j'，说明你切换了工作空间)
    name = "nlp"  # 图数据库的database 名称字符串
    # name = "mlcv"  # 图数据库的database 名称字符串

    # profile = "bolt://118.31.246.133:7687"  # 连接图数据库的url
    # name = "neo4j"  # 图数据库的database 名称字符串

    auth = ("neo4j", "Lzx19960802")  # 用户名和密码
    # 实例化连接对象
    graph = Graph(profile=profile, name=name, auth=auth)

    # 第二步：读取本地 CSV 文件
    # sep=",": 指定列之间的分隔符是逗号
    # header=None: 告诉 Pandas 这个文件没有表头（第一行就是真实数据，而不是列名）
    # df.values 将会是一个二维的 Numpy 数组，例如: [[1, '周杰伦', 2000, 44], [2, '林俊杰', 2003, 42]]
    df = pd.read_csv("./artists3.csv", sep=",", header=None)

    # 第三步：编写高级的 Upsert (更新/插入) Cypher 语句
    cql_str = """
        // 1. 根据唯一标识符 id 来匹配或创建节点。注意用 toInteger() 把字符串转为数字
        MERGE (a:Artist {id:toInteger($id)})
        // 2. ON CREATE SET: 如果上面的节点是刚刚【新创建】的，就执行下面的赋值
        ON CREATE SET
            a.id = toInteger($id),
            a.name = $name,             // 名字只有在创建时才写入
            a.year = toInteger($year),
            a.age = toInteger($age),
            a.created = timestamp()     // timestamp() 是 Neo4j 内置函数，生成当前毫秒级时间戳
        // 3. ON MATCH SET: 如果上面的节点是已经在库里【找到了】的，就执行下面的更新
        ON MATCH SET
            a.year = toInteger($year),  // 更新年份
            a.age = toInteger($age)     // 更新年龄 (注意：这里没有更新 name 和 created 字段)
        // 4. 返回处理后的节点信息
        RETURN a;
    """

    # 第四步：遍历数据并执行写入
    # 遍历 Pandas 读取出来的每一行数据
    for line in df.values:
        # line[0] 是 id, line[1] 是 name, line[2] 是 year, line[3] 是 age
        # 动态将这些参数传给 Cypher 语句中的 $变量
        r1 = graph.run(cql_str, id=line[0], name=line[1], year=line[2], age=line[3])
        # 打印返回的节点数据，确认写入成功
        print(r1.data())
