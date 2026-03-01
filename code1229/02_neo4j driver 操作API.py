# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/29 20:34
Create User : 19410
Desc : xxx
"""

# 导入官方驱动库的核心组件
# GraphDatabase: 用于创建数据库驱动实例
# Driver: 驱动对象的类型提示
# RoutingControl: 用于控制查询是发往“主节点(写)”还是“从节点(读)”，这是集群部署的关键
from neo4j import GraphDatabase, Driver, RoutingControl

"""
添加好友关系的函数
:param driver: 数据库驱动对象
:param db: 数据库名称
:param name1: 核心人物
:param name2: 好友名字
"""
def add_friend(driver: Driver, db: str, name1: str, name2: str):
    # execute_query 是官方驱动推荐的现代 API (Neo4j 4.x/5.x 引入)，它会自动管理事务
    driver.execute_query(
        """
            // MERGE 是图数据库里的神兵利器！相当于“如果存在就匹配，不存在就创建 (Upsert)”
            // 1. 确保名叫 name1 的人存在 (没有就建一个)
            MERGE (a:Person {name: $name1})
            // 2. 确保名叫 name2 的人存在 (没有就建一个)
            MERGE (b:Person {name: $name2})
            // 3. 确保 a 到 b 之间有一条叫 FRIEND 的关系 (没有就连一条)
            MERGE (a) -[:FRIEND]-> (b)
            // 4. 确保 b 到 a 之间也有一条 FRIEND 关系 (建立双向好友)
            MERGE (b) -[:FRIEND]-> (a)
        """,
        # 传入参数字典，防止 Cypher 注入
        name1=name1, name2=name2, database_=db
    )


# 查询并打印某人所有好友的函数
def print_friends(driver: Driver, db: str, name: str):
    records, summary, keys = driver.execute_query(
        """
            // 匹配以 a 为起点，通过 FRIEND 关系指向 friend 的节点
            // 并且限定起点 a 的名字必须是我们传入的参数 $name
            MATCH (a:Person)-[:FRIEND]->(friend:Person) WHERE a.name = $name 
            // 提取好友的名字，并重命名为 name
            RETURN friend.name as name
            // 按照拼音/字母顺序排序
            ORDER BY friend.name
        """,
        name=name, database_=db,
        # ⚠️ 高级特性：显式声明这是一个“只读”操作。
        # 如果你连接的是 Neo4j 集群，驱动会自动把这个查询路由到 Read Replica(只读从库) 上，减轻主库压力！
        routing_=RoutingControl.READ  # 当前是查询操作，默认该参数为 WRITE
    )
    print("查询执行摘要:", summary)  # 包含语句执行耗时、影响的节点数等底层信息
    print("返回的列名:", keys)  # 输出: ['name']

    # 遍历返回的记录集
    for record in records:
        print(type(record))  # 输出 <class 'neo4j._data.Record'>
        print(record)  # 输出 Record 对象本身
        print(record['name'])  # 像用字典一样，通过 RETURN 里的别名提取具体数据


if __name__ == '__main__':
    # neo4j:// 协议支持集群路由，bolt:// 是直连单机节点。本地测试通常用 bolt
    # url = "neo4j://127.0.0.1:7687"
    url = "neo4j://127.0.0.1:7687"  # 连接图数据库的url
    url = "bolt://127.0.0.1:7687"  # 连接图数据库的url
    db = "neo4j"  # 图数据库的database 名称字符串
    auth = ("neo4j", "Lzx19960802")  # 用户名和密码

    # ⚠️ 最佳实践：使用 with 语句上下文管理器！
    # 建立网络连接是很消耗资源的。with 代码块执行完毕后，无论有没有报错，都会自动安全地关闭 driver 连接
    # noinspection PyArgumentList
    with GraphDatabase.driver(url, auth=auth) as driver:
        # 构建社交网络拓扑：
        # 小明认识 -> 小红、小华、小沪
        # 小华认识 -> 张三
        # 张三认识 -> 王五
        add_friend(driver, db, "小明", "小红")
        add_friend(driver, db, "小明", "小华")
        add_friend(driver, db, "小明", "小沪")
        add_friend(driver, db, "小华", "张三")
        add_friend(driver, db, "张三", "王五")

        # 测试查询：打印小明的好友
        print("======== 小明的好友列表 ========")
        print_friends(driver, db, "小明")
