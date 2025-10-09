"""
Qdrant 完全指南 - AI原生向量数据库

Qdrant是一个用Rust编写的开源向量数据库和向量搜索引擎，
提供快速、可扩展的向量相似度搜索服务和便捷的API。

特点：
1. 🚀 高性能 - Rust编写，速度极快
2. 🔍 向量搜索 - 支持多种距离度量
3. 🎯 精确过滤 - 强大的payload过滤功能
4. 📦 易于部署 - Docker一键启动
5. 🌐 多语言SDK - Python、JavaScript、Rust等

作者: Marimo Notebook
日期: 2025-01-XX
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="Qdrant 完全指南")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🔍 Qdrant 完全指南

    ## 什么是Qdrant？

    **Qdrant** (读作: quadrant) 是一个AI原生的向量数据库和语义搜索引擎。它可以帮助你从非结构化数据中提取有意义的信息。

    ### 核心特性

    1. **高性能** ⚡
       - 用Rust编写，性能卓越
       - 支持HNSW索引算法
       - 毫秒级查询响应

    2. **向量搜索** 🔍
       - 支持余弦、点积、欧氏距离
       - 多向量支持
       - 稀疏向量支持

    3. **强大过滤** 🎯
       - 复杂的payload过滤
       - 地理位置搜索
       - 全文搜索

    4. **易于使用** 📦
       - RESTful API
       - gRPC API
       - 多语言SDK

    5. **可扩展** 🌐
       - 分布式部署
       - 水平扩展
       - 高可用性
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📦 安装和部署

    ### 方式1: Docker部署（推荐）

    ```bash
    # 拉取镜像
    docker pull qdrant/qdrant

    # 运行容器
    docker run -p 6333:6333 -p 6334:6334 \\
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \\
        qdrant/qdrant
    ```

    **访问地址：**
    - REST API: `http://localhost:6333`
    - Web UI: `http://localhost:6333/dashboard`
    - gRPC API: `http://localhost:6334`

    ### 方式2: Python本地模式

    ```bash
    pip install qdrant-client
    ```

    ```python
    from qdrant_client import QdrantClient

    # 本地内存模式（适合测试）
    client = QdrantClient(":memory:")

    # 或本地文件模式
    client = QdrantClient(path="./qdrant_data")
    ```

    ### 方式3: Qdrant Cloud

    - 免费层可用
    - 自动扩展
    - 提供Web UI
    - 无需维护基础设施

    访问: https://cloud.qdrant.io/
    """
    )
    return


@app.cell
def _():
    # 📦 步骤1: 导入必要的库
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import numpy as np

    print("=" * 60)
    print("🔍 Qdrant 客户端导入成功")
    print("=" * 60)

    # 📦 步骤2: 创建客户端连接
    # 使用内存模式（适合演示和测试）
    client = QdrantClient(":memory:")
    print("✅ 已连接到Qdrant（内存模式）")
    print("\n💡 提示: 内存模式数据不会持久化，重启后数据会丢失")
    print("   生产环境请使用: QdrantClient('localhost', port=6333)")

    return Distance, PointStruct, VectorParams, client, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎯 核心概念

    ### 1. Collection（集合）

    Collection是存储向量数据的命名空间，类似于数据库中的表。

    **特点：**
    - 每个collection有独立的配置
    - 支持多个向量字段
    - 可以设置不同的距离度量

    ### 2. Point（点）

    Point是Qdrant中的基本数据单元，包含：
    - **ID**: 唯一标识符
    - **Vector**: 向量数据
    - **Payload**: 附加元数据（可选）

    ### 3. Payload（负载）

    Payload是与向量关联的结构化数据，可以是：
    - 字符串、数字、布尔值
    - 数组、对象
    - 地理位置
    - 用于过滤和检索

    ### 4. Distance Metrics（距离度量）

    - **Cosine**: 余弦相似度（推荐用于归一化向量）
    - **Dot**: 点积（适合已归一化的向量）
    - **Euclid**: 欧氏距离（适合坐标数据）
    - **Manhattan**: 曼哈顿距离
    """
    )
    return


@app.cell
def _(Distance, VectorParams, client):
    print("=" * 60)
    print("📚 创建Collection")
    print("=" * 60)

    # 创建一个collection
    collection_name = "my_collection"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=128,  # 向量维度
            distance=Distance.COSINE  # 距离度量
        )
    )

    print(f"✅ Collection '{collection_name}' 创建成功")

    # 查看collection信息
    collection_info = client.get_collection(collection_name)
    print(f"\nCollection信息:")
    print(f"  - 向量维度: {collection_info.config.params.vectors.size}")
    print(f"  - 距离度量: {collection_info.config.params.vectors.distance}")
    print(f"  - 点数量: {collection_info.points_count}")
    return (collection_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📝 基础操作

    ### 1. 插入数据（Upsert）

    ```python
    from qdrant_client.models import PointStruct

    client.upsert(
        collection_name="my_collection",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, ...],
                payload={"city": "Berlin", "price": 100}
            ),
            # 更多点...
        ]
    )
    ```

    ### 2. 搜索（Query）

    ```python
    results = client.query_points(
        collection_name="my_collection",
        query=[0.1, 0.2, ...],
        limit=5
    ).points
    ```

    ### 3. 过滤搜索

    ```python
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    results = client.query_points(
        collection_name="my_collection",
        query=[0.1, 0.2, ...],
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="city",
                    match=MatchValue(value="Berlin")
                )
            ]
        ),
        limit=5
    ).points
    ```
    """
    )
    return


@app.cell
def _(PointStruct, client, collection_name, np):
    print("=" * 60)
    print("📝 插入向量数据")
    print("=" * 60)

    # 生成示例数据
    np.random.seed(42)
    n_points = 100

    # 创建points
    points = []
    cities = ["北京", "上海", "广州", "深圳", "杭州"]
    categories = ["电子产品", "服装", "食品", "图书"]

    for point_id in range(n_points):
        points.append(
            PointStruct(
                id=point_id,
                vector=np.random.rand(128).tolist(),
                payload={
                    "city": np.random.choice(cities),
                    "category": np.random.choice(categories),
                    "price": float(np.random.randint(10, 1000)),
                    "rating": float(np.random.uniform(3.0, 5.0)),
                    "name": f"产品_{point_id}"
                }
            )
        )

    # 批量插入
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )

    print(f"✅ 成功插入 {n_points} 个点")
    print(f"操作状态: {operation_info.status}")

    # 查看更新后的collection信息
    collection_info_updated = client.get_collection(collection_name)
    print(f"\n当前点数量: {collection_info_updated.points_count}")
    return


@app.cell
def _(client, collection_name, np):
    print("=" * 60)
    print("🔍 向量搜索")
    print("=" * 60)

    # 生成查询向量
    query_vector = np.random.rand(128).tolist()

    # 执行搜索（使用新的query_points API）
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,
        with_payload=True,
        with_vectors=False
    ).points

    print(f"找到 {len(search_results)} 个最相似的结果:\n")

    for idx, result in enumerate(search_results, 1):
        print(f"{idx}. ID: {result.id}")
        print(f"   相似度分数: {result.score:.4f}")
        print(f"   产品名称: {result.payload.get('name')}")
        print(f"   城市: {result.payload.get('city')}")
        print(f"   类别: {result.payload.get('category')}")
        print(f"   价格: ¥{result.payload.get('price'):.2f}")
        print(f"   评分: {result.payload.get('rating'):.2f}⭐")
        print()

    return query_vector, search_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎯 高级过滤

    Qdrant提供强大的过滤功能，支持复杂的查询条件。

    ### 过滤条件类型

    1. **Must**: 所有条件必须满足（AND）
    2. **Should**: 至少一个条件满足（OR）
    3. **Must Not**: 所有条件都不满足（NOT）

    ### 常用过滤器

    ```python
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, Range
    )

    # 精确匹配
    FieldCondition(key="city", match=MatchValue(value="Berlin"))

    # 范围查询
    FieldCondition(key="price", range=Range(gte=100, lte=500))

    # 多值匹配
    FieldCondition(key="category", match=MatchAny(any=["A", "B"]))

    # 地理位置
    FieldCondition(
        key="location",
        geo_radius=GeoRadius(
            center=GeoPoint(lon=13.4, lat=52.5),
            radius=1000.0  # 米
        )
    )
    ```
    """
    )
    return


@app.cell
def _(client, collection_name, np):
    print("=" * 60)
    print("🎯 带过滤的搜索")
    print("=" * 60)

    # 导入过滤相关的模型
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    # 生成查询向量
    query_vec = np.random.rand(128).tolist()

    # 示例1: 单条件过滤
    print("1️⃣ 搜索北京的产品:")
    results_1 = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        query_filter=Filter(
            must=[
                FieldCondition(key="city", match=MatchValue(value="北京"))
            ]
        ),
        limit=3,
        with_payload=True
    ).points

    for item in results_1:
        print(f"  - {item.payload['name']} | {item.payload['city']} | ¥{item.payload['price']}")

    # 示例2: 多条件过滤
    print("\n2️⃣ 搜索价格在100-500之间的电子产品:")
    results_2 = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        query_filter=Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value="电子产品")),
                FieldCondition(key="price", range=Range(gte=100, lte=500))
            ]
        ),
        limit=3,
        with_payload=True
    ).points

    for item in results_2:
        print(f"  - {item.payload['name']} | {item.payload['category']} | ¥{item.payload['price']}")

    # 示例3: 复杂过滤
    print("\n3️⃣ 搜索评分>4.0且价格<300的产品:")
    results_3 = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        query_filter=Filter(
            must=[
                FieldCondition(key="rating", range=Range(gt=4.0)),
                FieldCondition(key="price", range=Range(lt=300))
            ]
        ),
        limit=3,
        with_payload=True
    ).points

    for item in results_3:
        print(f"  - {item.payload['name']} | 评分:{item.payload['rating']:.2f} | ¥{item.payload['price']}")

    return Filter, FieldCondition, MatchValue, Range, query_vec, results_1, results_2, results_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📊 Payload操作

    ### 1. 设置Payload

    ```python
    # 为单个点设置payload
    client.set_payload(
        collection_name="my_collection",
        payload={"new_field": "value"},
        points=[1, 2, 3]
    )
    ```

    ### 2. 删除Payload字段

    ```python
    client.delete_payload(
        collection_name="my_collection",
        keys=["field_to_delete"],
        points=[1, 2, 3]
    )
    ```

    ### 3. 清空Payload

    ```python
    client.clear_payload(
        collection_name="my_collection",
        points=[1, 2, 3]
    )
    ```

    ### 4. 创建Payload索引

    为了加速过滤查询，建议为常用字段创建索引：

    ```python
    client.create_payload_index(
        collection_name="my_collection",
        field_name="city",
        field_schema="keyword"
    )
    ```
    """
    )
    return


@app.cell
def _(client, collection_name):
    print("=" * 60)
    print("📊 Payload索引")
    print("=" * 60)

    print("⚠️  注意: 内存模式不支持payload索引")
    print("   如需使用索引功能，请连接到Qdrant服务器:")
    print("   client = QdrantClient('localhost', port=6333)")
    print()

    # 为常用字段创建索引（仅在服务器模式下有效）
    fields_to_index = ["city", "category", "price", "rating"]

    print("💡 在服务器模式下，可以为以下字段创建索引:")
    for field in fields_to_index:
        if field in ["city", "category"]:
            schema_type = "keyword"
        else:
            schema_type = "float"
        print(f"   - {field} (类型: {schema_type})")

    print("\n📝 创建索引的代码示例:")
    print("""
    client.create_payload_index(
        collection_name="my_collection",
        field_name="city",
        field_schema="keyword"
    )
    """)

    # 查看collection的索引信息
    collection_info_with_index = client.get_collection(collection_name)
    print(f"当前索引数量: {len(collection_info_with_index.payload_schema)}")

    return fields_to_index, collection_info_with_index


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔄 批量操作

    ### 1. 批量插入

    ```python
    # 使用batch方法提高性能
    from qdrant_client.models import Batch

    client.upsert(
        collection_name="my_collection",
        points=Batch(
            ids=[1, 2, 3, ...],
            vectors=[[...], [...], [...]],
            payloads=[{...}, {...}, {...}]
        )
    )
    ```

    ### 2. 滚动查询（Scroll）

    用于遍历大量数据：

    ```python
    # 获取所有点
    records, next_page = client.scroll(
        collection_name="my_collection",
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    ```

    ### 3. 批量删除

    ```python
    # 按ID删除
    client.delete(
        collection_name="my_collection",
        points_selector=[1, 2, 3, 4, 5]
    )

    # 按过滤条件删除
    client.delete(
        collection_name="my_collection",
        points_selector=Filter(
            must=[FieldCondition(key="city", match=MatchValue(value="Berlin"))]
        )
    )
    ```
    """
    )
    return


@app.cell
def _(client, collection_name):
    print("=" * 60)
    print("🔄 滚动查询示例")
    print("=" * 60)

    # 使用scroll获取所有点
    all_points = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=20,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        all_points.extend(records)

        if next_offset is None:
            break
        offset = next_offset

    print(f"✅ 通过scroll获取了 {len(all_points)} 个点")

    # 显示前5个点的信息
    print("\n前5个点的信息:")
    for pt in all_points[:5]:
        print(f"  - ID: {pt.id} | {pt.payload['name']} | {pt.payload['city']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎨 高级搜索功能

    ### 1. 推荐搜索（Recommend）

    基于正负样本进行推荐：

    ```python
    client.recommend(
        collection_name="my_collection",
        positive=[1, 2, 3],  # 喜欢的样本
        negative=[4, 5],     # 不喜欢的样本
        limit=10
    )
    ```

    ### 2. 批量搜索

    ```python
    client.search_batch(
        collection_name="my_collection",
        requests=[
            SearchRequest(vector=[...], limit=5),
            SearchRequest(vector=[...], limit=5),
        ]
    )
    ```

    ### 3. 分组搜索

    按某个字段分组返回结果：

    ```python
    client.search_groups(
        collection_name="my_collection",
        query_vector=[...],
        group_by="city",
        limit=3,
        group_size=2
    )
    ```

    ### 4. 发现搜索（Discover）

    在向量空间中探索：

    ```python
    client.discover(
        collection_name="my_collection",
        target=1,  # 目标点
        context=[
            (2, 3),  # 正样本对
            (4, 5),  # 负样本对
        ],
        limit=10
    )
    ```
    """
    )
    return


@app.cell
def _(client, collection_name):
    print("=" * 60)
    print("🎨 推荐搜索示例")
    print("=" * 60)

    # 选择一些正样本和负样本
    positive_ids = [0, 1, 2]
    negative_ids = [50, 51]

    # 执行推荐搜索
    recommendations = client.recommend(
        collection_name=collection_name,
        positive=positive_ids,
        negative=negative_ids,
        limit=5,
        with_payload=True
    )

    print(f"基于正样本 {positive_ids} 和负样本 {negative_ids} 的推荐结果:\n")

    for num, rec in enumerate(recommendations, 1):
        print(f"{num}. {rec.payload['name']}")
        print(f"   城市: {rec.payload['city']} | 类别: {rec.payload['category']}")
        print(f"   分数: {rec.score:.4f}")
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔧 Collection管理

    ### 1. 列出所有Collections

    ```python
    collections = client.get_collections()
    for collection in collections.collections:
        print(collection.name)
    ```

    ### 2. 删除Collection

    ```python
    client.delete_collection("my_collection")
    ```

    ### 3. 创建Collection别名

    ```python
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAlias(
                create_alias=CreateAliasOperation(
                    collection_name="my_collection",
                    alias_name="my_alias"
                )
            )
        ]
    )
    ```

    ### 4. 更新Collection配置

    ```python
    client.update_collection(
        collection_name="my_collection",
        optimizer_config=OptimizersConfigDiff(
            indexing_threshold=10000
        )
    )
    ```

    ### 5. Collection快照

    ```python
    # 创建快照
    snapshot_info = client.create_snapshot(
        collection_name="my_collection"
    )

    # 恢复快照
    client.recover_snapshot(
        collection_name="my_collection",
        snapshot_path="/path/to/snapshot"
    )
    ```
    """
    )
    return


@app.cell
def _(client):
    print("=" * 60)
    print("🔧 Collection管理")
    print("=" * 60)

    # 列出所有collections
    collections_list = client.get_collections()

    print(f"当前有 {len(collections_list.collections)} 个collection:\n")

    for collection in collections_list.collections:
        print(f"📚 {collection.name}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🚀 性能优化

    ### 1. 向量量化

    减少内存使用和提高搜索速度：

    ```python
    from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig

    client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type="int8",
                quantile=0.99,
                always_ram=True
            )
        )
    )
    ```

    ### 2. HNSW索引配置

    ```python
    from qdrant_client.models import HnswConfigDiff

    client.update_collection(
        collection_name="my_collection",
        hnsw_config=HnswConfigDiff(
            m=16,  # 每层的连接数
            ef_construct=100,  # 构建时的搜索深度
        )
    )
    ```

    ### 3. 优化器配置

    ```python
    from qdrant_client.models import OptimizersConfigDiff

    client.update_collection(
        collection_name="my_collection",
        optimizer_config=OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=50000
        )
    )
    ```

    ### 4. 批量操作

    - 使用`wait=False`进行异步操作
    - 批量插入时使用`Batch`
    - 合理设置batch大小（1000-10000）
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🌐 实战案例：语义搜索

    ### 场景：构建产品语义搜索系统

    ```python
    from sentence_transformers import SentenceTransformer

    # 1. 加载embedding模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. 创建collection
    client.create_collection(
        collection_name="products",
        vectors_config=VectorParams(
            size=384,  # all-MiniLM-L6-v2的维度
            distance=Distance.COSINE
        )
    )

    # 3. 准备数据
    products = [
        {"id": 1, "name": "iPhone 15", "description": "最新款苹果手机"},
        {"id": 2, "name": "MacBook Pro", "description": "专业笔记本电脑"},
        # ...
    ]

    # 4. 生成embeddings并插入
    points = []
    for product in products:
        text = f"{product['name']} {product['description']}"
        vector = model.encode(text).tolist()

        points.append(PointStruct(
            id=product['id'],
            vector=vector,
            payload=product
        ))

    client.upsert(collection_name="products", points=points)

    # 5. 语义搜索
    query = "我想买一台笔记本"
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name="products",
        query=query_vector,
        limit=5
    ).points
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 最佳实践

    ### 1. Collection设计

    - ✅ 为不同类型的数据创建不同的collection
    - ✅ 合理设置向量维度（不要过大）
    - ✅ 选择合适的距离度量
    - ✅ 为常用过滤字段创建索引

    ### 2. 数据插入

    - ✅ 使用批量操作提高效率
    - ✅ 合理设置batch大小
    - ✅ 使用`wait=False`进行异步操作
    - ✅ 定期优化collection

    ### 3. 搜索优化

    - ✅ 使用payload索引加速过滤
    - ✅ 只返回需要的字段
    - ✅ 合理设置limit
    - ✅ 使用量化减少内存

    ### 4. 生产部署

    - ✅ 使用持久化存储
    - ✅ 配置备份策略
    - ✅ 监控性能指标
    - ✅ 考虑分布式部署

    ### 5. 安全性

    - ✅ 启用API密钥认证
    - ✅ 使用TLS加密
    - ✅ 限制网络访问
    - ✅ 定期更新版本
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔗 资源链接

    ### 官方资源

    - 🌐 [官方网站](https://qdrant.tech/)
    - 📚 [文档](https://qdrant.tech/documentation/)
    - 🐙 [GitHub](https://github.com/qdrant/qdrant)
    - 🐍 [Python客户端](https://github.com/qdrant/qdrant-client)
    - ☁️ [Qdrant Cloud](https://cloud.qdrant.io/)

    ### 学习资源

    - [快速开始教程](https://qdrant.tech/documentation/quickstart/)
    - [向量搜索基础](https://qdrant.tech/documentation/beginner-tutorials/)
    - [API参考](https://qdrant.tech/documentation/api-reference/)
    - [示例项目](https://github.com/qdrant/examples)

    ### 社区

    - [Discord](https://qdrant.to/discord)
    - [Twitter](https://twitter.com/qdrant_engine)
    - [LinkedIn](https://www.linkedin.com/company/qdrant/)
    - [YouTube](https://www.youtube.com/channel/UC6ftm8PwH1RU_LM1jwG0LQA)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📝 总结

    ### Qdrant的核心优势

    1. **高性能** - Rust编写，速度极快
    2. **易用性** - 简洁的API，丰富的SDK
    3. **功能强大** - 支持复杂过滤和多种搜索模式
    4. **可扩展** - 支持分布式部署和水平扩展
    5. **开源免费** - MIT许可证，社区活跃

    ### 适用场景

    - ✅ 语义搜索
    - ✅ 推荐系统
    - ✅ 图像搜索
    - ✅ 问答系统
    - ✅ 异常检测
    - ✅ RAG应用

    ### 何时使用Qdrant

    - 需要高性能向量搜索
    - 需要复杂的过滤功能
    - 需要可扩展的解决方案
    - 需要开源和自托管选项
    - 构建AI应用

    ### 开始使用

    ```bash
    # 1. 安装客户端
    pip install qdrant-client

    # 2. 启动服务
    docker run -p 6333:6333 qdrant/qdrant

    # 3. 开始编码
    from qdrant_client import QdrantClient
    client = QdrantClient("localhost", port=6333)
    ```

    ---

    **Qdrant = 高性能 + 易用性 + 强大功能** 🔍🚀
    """
    )
    return


if __name__ == "__main__":
    app.run()
