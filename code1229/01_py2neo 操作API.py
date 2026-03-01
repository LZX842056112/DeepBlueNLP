# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/29 20:17
Create User : 19410
Desc : xxx
"""

if __name__ == '__main__':
    # 从 py2neo 库导入 Graph（用于连接数据库）和 Cursor（用于接收查询结果）
    from py2neo import Graph
    from py2neo.cypher import Cursor

    # 第一步：配置连接参数并连接 Neo4j 数据库
    # profile: 数据库的地址。bolt 是 Neo4j 专用的高速传输协议，默认端口 7687
    profile = "bolt://127.0.0.1:7687"  # 连接图数据库的url
    # name: 指定你要操作的具体是哪个数据库实例（类似于 MySQL 里的 use database）
    name = "nlp"  # 图数据库的database 名称字符串
    # name = "mlcv"  # 图数据库的database 名称字符串

    # profile = "bolt://118.31.246.133:7687"  # 连接图数据库的url
    # name = "neo4j"  # 图数据库的database 名称字符串

    # auth: 元组形式的 (账号, 密码)。默认账号通常是 neo4j
    auth = ("neo4j", "Lzx19960802")  # 用户名和密码
    # 建立与图数据库的连接，实例化 Graph 对象
    graph = Graph(profile=profile, name=name, auth=auth)

    # 第二步：执行 CREATE (插入数据)
    # graph.run() 用于执行原生的 Cypher 查询语句（Neo4j的SQL）
    r1: Cursor = graph.run(
        """
            // Cypher语法: ()代表节点。这里创建了一个标签为 Book 的节点。
            // 别名为 b，并且将传入的参数 $name 和 $book_price 赋给它的属性。
            CREATE (b:Book {name:$name, price:$book_price}) RETURN b
        """,
        # 这里的参数会动态替换掉上面语句里的 $name 和 $book_price，防止注入攻击
        name='深度学习基础',
        book_price=13.52
    )
    print(type(r1))  # 结果是一个 Cursor (游标对象)
    print(r1)  # 打印游标的内存地址等信息
    print(r1.data())  # 把游标里的数据全部取出来，转成 Python 的 List[Dict] 格式

    # 第三步：执行 MATCH (简单查询) 并转为 DataFrame
    # MATCH 相当于 SQL 里的 SELECT。这里匹配所有 name 为 '深度学习基础' 的 Book 节点
    # RUN方法执行结果是一个迭代器
    r2 = graph.run("MATCH (b:Book {name:$name}) RETURN b.name AS name, b.price AS price", name='深度学习基础')
    # 非常实用的功能：直接将查询结果转成 pandas 的 DataFrame，方便后续做数据分析！
    print(r2.to_data_frame())
    # print(r2.to_ndarray())

    # 第四步：执行复杂的路径查询 (推荐算法核心) 并手动迭代
    r3: Cursor = graph.run("""
        // 1. 找合作演员: 匹配成龙参演过的电影，以及这些电影里的其他参演人员 (p2)
        // 语法解析: (节点)-[关系]->(节点)
        MATCH
            (p1:Person {name:'成龙'})-[:参演]->(moive)<-[:参演]-(p2:Person)
        // 2. WITH 相当于管道符，把上面找到的 p2 (成龙的合作演员) 传递给下一步
        WITH p2
        // 3. 找合作演员的电影: 查找这些合作演员 (p2) 都参演过哪些其他电影 (moive2)
        MATCH
            (p2)-[:参演]->(moive2)
        // 4. 返回 演员 和 电影 节点
        RETURN p2,moive2;
    """)
    # 使用 while 循环配合 r3.forward()，逐条(一行一行)读取数据
    # 这种方式比 r3.data() 更省内存，适合返回上万条数据的大查询
    while r3.forward():
        print("=" * 100)
        # r3.current 获取游标当前指向的这一行记录 (Record)
        current_record = r3.current
        print(type(current_record))
        print(current_record)
        # 通过字典键值对的方式，提取 RETURN 语句里定义的别名 'p2' 和 'moive2'
        print(current_record['p2'])  # 打印 Person 节点对象
        print(current_record['moive2'])  # 打印 moive2 节点对象
        # 获取节点的具体属性值 (比如电影名)
        print(current_record['moive2']['name'])
        # 推荐使用 .get() 方法获取属性，因为如果该节点没有这个属性，不会报错而是返回 None
        print(current_record['moive2'].get('name'))
        print(current_record['p2'].get('occupation'))  # 尝试获取职业属性
