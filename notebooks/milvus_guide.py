import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🗄️ Milvus向量数据库完全指南

    Milvus是一个开源的向量数据库，专为AI应用和向量相似度搜索而设计。

    ## 🎯 为什么使用Milvus？

    - **高性能**: 支持十亿级向量的毫秒级搜索
    - **可扩展**: 支持水平扩展和分布式部署
    - **多种索引**: 支持IVF、HNSW、DiskANN等多种索引算法
    - **混合搜索**: 支持向量搜索+标量过滤
    - **云原生**: 基于Kubernetes的云原生架构

    ## 📦 安装

    ```bash
    # 安装Python SDK
    pip install pymilvus

    # 或使用uv
    uv pip install pymilvus
    ```

    ## 🐳 启动Milvus服务

    ```bash
    # 使用Docker Compose启动（推荐）
    wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
    docker-compose up -d

    # 或使用Milvus Lite（嵌入式版本）
    pip install milvus
    ```

    当前版本要求: `pymilvus>=2.3.0`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣ 基础连接

    ### 连接到Milvus服务器

    Milvus支持多种连接方式：
    - 本地单机版（Standalone）
    - 集群版（Cluster）
    - Milvus Lite（嵌入式）
    """
    )
    return


@app.cell
def _():
    from pymilvus import connections, utility
    import numpy as np

    print("=" * 60)
    print("🔌 连接到Milvus")
    print("=" * 60)

    # 连接到本地Milvus服务器
    # 默认地址: localhost:19530
    try:
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
        print("\n✅ 成功连接到Milvus服务器")

        # 查看服务器版本
        print(f"📌 Milvus版本: {utility.get_server_version()}")

    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print("\n💡 提示:")
        print("   1. 确保Milvus服务已启动")
        print("   2. 检查端口19530是否可访问")
        print("   3. 或使用Milvus Lite: pip install milvus")

    return np, utility


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2️⃣ Collection（集合）管理

    ### Collection基本概念

    Collection类似于关系数据库中的表，是Milvus中存储数据的基本单位。

    ### 主要操作

    | 操作 | 函数 | 说明 |
    |------|------|------|
    | 创建 | `Collection()` | 创建新集合 |
    | 列出 | `utility.list_collections()` | 列出所有集合 |
    | 检查 | `utility.has_collection()` | 检查集合是否存在 |
    | 删除 | `utility.drop_collection()` | 删除集合 |
    | 加载 | `collection.load()` | 加载到内存 |
    | 释放 | `collection.release()` | 从内存释放 |
    """
    )
    return


@app.cell
def _(utility):
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

    print("=" * 60)
    print("📚 Collection管理")
    print("=" * 60)

    # 定义字段Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="score", dtype=DataType.FLOAT)
    ]

    # 创建Collection Schema
    schema = CollectionSchema(
        fields=fields,
        description="示例集合"
    )

    collection_name = "demo_collection"

    # 删除已存在的集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"\n🗑️  已删除旧集合: {collection_name}")

    # 创建新集合
    collection = Collection(
        name=collection_name,
        schema=schema
    )

    print(f"\n✅ 创建集合: {collection_name}")
    print(f"📊 字段数量: {len(fields)}")
    print(f"📏 向量维度: 128")

    # 列出所有集合
    print(f"\n📋 所有集合: {utility.list_collections()}")

    return Collection, CollectionSchema, DataType, FieldSchema, collection


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3️⃣ 数据插入

    ### 插入数据的方式

    | 方式 | 说明 | 适用场景 |
    |------|------|---------|
    | `insert()` | 批量插入 | 大量数据导入 |
    | `upsert()` | 插入或更新 | 数据更新场景 |

    ### 数据格式

    - 列表格式：`[[id1, id2, ...], [vec1, vec2, ...], ...]`
    - 字典格式：`[{"id": 1, "embedding": [...], ...}, ...]`
    """
    )
    return


@app.cell
def _(collection, np):
    print("=" * 60)
    print("📥 插入数据")
    print("=" * 60)

    # 生成示例数据
    num_entities = 1000

    # 生成随机向量
    embeddings = np.random.random((num_entities, 128)).tolist()

    # 生成其他字段数据
    ids = list(range(num_entities))
    texts = [f"文本_{i}" for i in range(num_entities)]
    scores = np.random.random(num_entities).tolist()

    # 准备插入数据
    entities = [
        ids,
        embeddings,
        texts,
        scores
    ]

    print(f"\n准备插入 {num_entities} 条数据...")

    # 插入数据
    insert_result = collection.insert(entities)

    print(f"✅ 插入成功")
    print(f"📊 插入数量: {insert_result.insert_count}")
    print(f"🔑 主键范围: {insert_result.primary_keys[:5]}... (显示前5个)")

    # 刷新数据（确保数据持久化）
    collection.flush()
    print(f"\n💾 数据已刷新到磁盘")

    # 查看集合统计
    print(f"\n📈 集合统计:")
    print(f"   总数据量: {collection.num_entities}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4️⃣ 索引管理

    ### 常用索引类型

    | 索引类型 | 说明 | 适用场景 |
    |---------|------|---------|
    | **FLAT** | 暴力搜索 | 小数据集，追求100%召回率 |
    | **IVF_FLAT** | 倒排文件 | 中等数据集，平衡性能和召回 |
    | **IVF_SQ8** | 标量量化 | 节省内存 |
    | **IVF_PQ** | 乘积量化 | 大数据集，节省内存 |
    | **HNSW** | 分层图 | 高性能，高召回率 |
    | **ANNOY** | 树结构 | 静态数据 |

    ### 索引参数

    不同索引类型有不同的参数配置。
    """
    )
    return


@app.cell
def _(collection):
    print("=" * 60)
    print("🔍 创建索引")
    print("=" * 60)

    # 定义索引参数
    index_params = {
        "metric_type": "L2",        # 距离度量：L2（欧氏距离）或IP（内积）
        "index_type": "IVF_FLAT",   # 索引类型
        "params": {"nlist": 128}    # 索引参数：聚类中心数量
    }

    print(f"\n索引配置:")
    print(f"  类型: {index_params['index_type']}")
    print(f"  度量: {index_params['metric_type']}")
    print(f"  参数: {index_params['params']}")

    # 在embedding字段上创建索引
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    print(f"\n✅ 索引创建成功")

    # 查看索引信息
    index_info = collection.index()
    print(f"\n📋 索引信息:")
    print(f"   字段: {index_info.field_name}")
    print(f"   类型: {index_info.params['index_type']}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5️⃣ 向量搜索

    ### 搜索类型

    | 类型 | 函数 | 说明 |
    |------|------|------|
    | 向量搜索 | `search()` | ANN近似最近邻搜索 |
    | 查询 | `query()` | 基于标量字段的精确查询 |
    | 混合搜索 | `search()` + `expr` | 向量搜索+标量过滤 |

    ### 搜索参数

    - `data`: 查询向量
    - `anns_field`: 向量字段名
    - `param`: 搜索参数
    - `limit`: 返回结果数量
    - `expr`: 过滤表达式
    - `output_fields`: 返回的字段
    """
    )
    return


@app.cell
def _(collection, np):
    print("=" * 60)
    print("🔎 向量搜索")
    print("=" * 60)

    # 加载集合到内存（搜索前必须）
    collection.load()
    print("\n📂 集合已加载到内存")

    # 生成查询向量
    search_vectors = np.random.random((3, 128)).tolist()

    # 定义搜索参数
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}  # 搜索的聚类数量
    }

    print(f"\n搜索配置:")
    print(f"  查询向量数: 3")
    print(f"  返回Top-K: 5")
    print(f"  搜索参数: nprobe=10")

    # 执行搜索
    results = collection.search(
        data=search_vectors,
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["text", "score"]
    )

    print(f"\n✅ 搜索完成")
    print(f"\n📊 搜索结果:")

    for i, hits in enumerate(results):
        print(f"\n查询 {i+1}:")
        for j, hit in enumerate(hits):
            print(f"  Top-{j+1}: ID={hit.id}, 距离={hit.distance:.4f}, "
                  f"文本={hit.entity.get('text')}, 分数={hit.entity.get('score'):.4f}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 6️⃣ 标量查询和过滤

    ### 查询表达式

    Milvus支持丰富的过滤表达式：

    | 操作符 | 说明 | 示例 |
    |--------|------|------|
    | `==` | 等于 | `id == 1` |
    | `!=` | 不等于 | `score != 0.5` |
    | `>`, `>=` | 大于、大于等于 | `score > 0.8` |
    | `<`, `<=` | 小于、小于等于 | `score < 0.5` |
    | `in` | 在列表中 | `id in [1, 2, 3]` |
    | `not in` | 不在列表中 | `id not in [1, 2]` |
    | `and`, `or` | 逻辑与、或 | `score > 0.5 and id < 100` |
    | `like` | 模糊匹配 | `text like "文本%"` |
    """
    )
    return


@app.cell
def _(collection):
    print("=" * 60)
    print("🔍 标量查询")
    print("=" * 60)

    # 1. 简单查询
    print("\n1️⃣ 查询ID在范围内的数据:")
    query_result1 = collection.query(
        expr="id in [0, 1, 2, 3, 4]",
        output_fields=["id", "text", "score"]
    )

    for item in query_result1[:3]:
        print(f"   ID={item['id']}, 文本={item['text']}, 分数={item['score']:.4f}")

    # 2. 条件查询
    print("\n2️⃣ 查询分数大于0.8的数据:")
    query_result2 = collection.query(
        expr="score > 0.8",
        output_fields=["id", "text", "score"],
        limit=5
    )

    print(f"   找到 {len(query_result2)} 条数据")
    for item in query_result2[:3]:
        print(f"   ID={item['id']}, 分数={item['score']:.4f}")

    # 3. 复合条件查询
    print("\n3️⃣ 复合条件查询 (score > 0.5 and id < 100):")
    query_result3 = collection.query(
        expr="score > 0.5 and id < 100",
        output_fields=["id", "text", "score"],
        limit=5
    )

    print(f"   找到 {len(query_result3)} 条数据")
    for item in query_result3[:3]:
        print(f"   ID={item['id']}, 分数={item['score']:.4f}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 7️⃣ 混合搜索

    ### 向量搜索 + 标量过滤

    混合搜索结合了向量相似度搜索和标量字段过滤，是实际应用中最常用的功能。
    """
    )
    return


@app.cell
def _(collection, np):
    print("=" * 60)
    print("🎯 混合搜索")
    print("=" * 60)

    # 生成查询向量
    _search_vec = np.random.random((1, 128)).tolist()

    # 混合搜索：向量搜索 + 分数过滤
    print("\n搜索条件: 向量相似度 + score > 0.7")

    _hybrid_results = collection.search(
        data=_search_vec,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        expr="score > 0.7",  # 标量过滤条件
        output_fields=["text", "score"]
    )

    print(f"\n✅ 找到 {len(_hybrid_results[0])} 条结果")
    print(f"\n📊 Top-5 结果:")

    for _idx, _hit in enumerate(_hybrid_results[0][:5]):
        print(f"  {_idx+1}. ID={_hit.id}, 距离={_hit.distance:.4f}, "
              f"分数={_hit.entity.get('score'):.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 8️⃣ 数据删除

    ### 删除操作

    | 操作 | 函数 | 说明 |
    |------|------|------|
    | 按表达式删除 | `delete()` | 根据条件删除数据 |
    | 删除集合 | `drop_collection()` | 删除整个集合 |

    ### 删除表达式

    使用与查询相同的表达式语法。
    """
    )
    return


@app.cell
def _(collection):
    print("=" * 60)
    print("🗑️  数据删除")
    print("=" * 60)

    # 查看删除前的数据量
    _count_before = collection.num_entities
    print(f"\n删除前数据量: {_count_before}")

    # 删除ID在指定范围内的数据
    _delete_expr = "id in [0, 1, 2, 3, 4]"
    print(f"\n删除条件: {_delete_expr}")

    _delete_result = collection.delete(_delete_expr)

    print(f"✅ 删除成功")
    print(f"📊 删除数量: {_delete_result.delete_count}")

    # 刷新以查看最新数据量
    collection.flush()
    _count_after = collection.num_entities
    print(f"\n删除后数据量: {_count_after}")
    print(f"实际减少: {_count_before - _count_after}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 9️⃣ 分区管理

    ### Partition（分区）

    分区是Collection的逻辑划分，可以提高查询效率。

    | 操作 | 函数 | 说明 |
    |------|------|------|
    | 创建分区 | `create_partition()` | 创建新分区 |
    | 列出分区 | `partitions` | 查看所有分区 |
    | 删除分区 | `drop_partition()` | 删除分区 |
    | 加载分区 | `load()` | 加载指定分区 |

    ### 使用场景

    - 按时间分区（如按月、按年）
    - 按类别分区（如按产品类型）
    - 按地域分区（如按国家、城市）
    """
    )
    return


@app.cell
def _(collection):
    print("=" * 60)
    print("📂 分区管理")
    print("=" * 60)

    # 创建分区
    _partition_name = "partition_2024"

    if not collection.has_partition(_partition_name):
        _partition = collection.create_partition(_partition_name)
        print(f"\n✅ 创建分区: {_partition_name}")
    else:
        _partition = collection.partition(_partition_name)
        print(f"\n📌 分区已存在: {_partition_name}")

    # 列出所有分区
    print(f"\n📋 所有分区:")
    for _p in collection.partitions:
        print(f"   - {_p.name} (数据量: {_p.num_entities})")

    # 向分区插入数据
    _partition_data = [
        [10000, 10001, 10002],  # IDs
        [[0.1] * 128, [0.2] * 128, [0.3] * 128],  # embeddings
        ["分区文本_1", "分区文本_2", "分区文本_3"],  # texts
        [0.9, 0.8, 0.7]  # scores
    ]

    _partition.insert(_partition_data)
    collection.flush()

    print(f"\n✅ 向分区插入 3 条数据")
    print(f"📊 分区数据量: {_partition.num_entities}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔟 实战案例：文本语义搜索

    ### 场景说明

    使用预训练的Sentence Transformer模型，实现文本的语义搜索。

    ### 步骤

    1. 加载预训练模型
    2. 将文本转换为向量
    3. 存储到Milvus
    4. 执行语义搜索
    """
    )
    return


@app.cell
def _(Collection, CollectionSchema, DataType, FieldSchema, np, utility):
    print("=" * 60)
    print("💼 实战案例：文本语义搜索")
    print("=" * 60)

    # 模拟文本嵌入（实际应用中使用Sentence Transformer）
    def get_text_embedding(text):
        """模拟文本嵌入函数"""
        # 实际应用中应该使用：
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # return model.encode(text)
        return np.random.random(384).tolist()

    # 示例文本数据
    _documents = [
        "Python是一种高级编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理处理文本数据",
        "计算机视觉处理图像数据",
        "数据科学结合统计学和编程",
        "云计算提供按需计算资源",
        "区块链是分布式账本技术",
        "物联网连接物理设备到互联网",
        "大数据处理海量数据集"
    ]

    print(f"\n📚 文档数量: {len(_documents)}")

    # 创建文本搜索集合
    _text_collection_name = "text_search_demo"

    if utility.has_collection(_text_collection_name):
        utility.drop_collection(_text_collection_name)

    _text_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]

    _text_schema = CollectionSchema(fields=_text_fields, description="文本搜索集合")
    _text_collection = Collection(name=_text_collection_name, schema=_text_schema)

    print(f"✅ 创建集合: {_text_collection_name}")

    # 生成嵌入并插入
    _embeddings_list = [get_text_embedding(doc) for doc in _documents]

    _text_entities = [
        _embeddings_list,
        _documents
    ]

    _text_collection.insert(_text_entities)
    _text_collection.flush()

    print(f"✅ 插入 {len(_documents)} 条文档")

    # 创建索引
    _text_index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 16}
    }

    _text_collection.create_index(field_name="embedding", index_params=_text_index_params)
    _text_collection.load()

    print(f"✅ 索引创建完成，集合已加载")

    # 执行语义搜索
    _query_text = "什么是人工智能"
    _query_embedding = [get_text_embedding(_query_text)]

    print(f"\n🔍 查询: '{_query_text}'")

    _search_results = _text_collection.search(
        data=_query_embedding,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=3,
        output_fields=["text"]
    )

    print(f"\n📊 Top-3 相关文档:")
    for _i, _hit in enumerate(_search_results[0]):
        print(f"  {_i+1}. {_hit.entity.get('text')} (距离: {_hit.distance:.4f})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 API速查表

    ### 连接管理

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `connections.connect()` | 连接服务器 | `connections.connect(host='localhost', port='19530')` |
    | `connections.disconnect()` | 断开连接 | `connections.disconnect('default')` |
    | `utility.get_server_version()` | 获取版本 | `utility.get_server_version()` |

    ### Collection操作

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `Collection()` | 创建/获取集合 | `Collection(name='demo', schema=schema)` |
    | `collection.insert()` | 插入数据 | `collection.insert(data)` |
    | `collection.search()` | 向量搜索 | `collection.search(data, anns_field, param, limit)` |
    | `collection.query()` | 标量查询 | `collection.query(expr, output_fields)` |
    | `collection.delete()` | 删除数据 | `collection.delete(expr)` |
    | `collection.load()` | 加载到内存 | `collection.load()` |
    | `collection.release()` | 释放内存 | `collection.release()` |

    ### 索引操作

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `create_index()` | 创建索引 | `collection.create_index(field_name, index_params)` |
    | `drop_index()` | 删除索引 | `collection.drop_index()` |
    | `index()` | 查看索引 | `collection.index()` |

    ### 工具函数

    | 函数 | 说明 | 示例 |
    |------|------|------|
    | `utility.list_collections()` | 列出集合 | `utility.list_collections()` |
    | `utility.has_collection()` | 检查集合 | `utility.has_collection('demo')` |
    | `utility.drop_collection()` | 删除集合 | `utility.drop_collection('demo')` |

    ## 💡 最佳实践

    ### 1. 选择合适的索引

    - **小数据集(<1M)**: FLAT
    - **中等数据集(1M-10M)**: IVF_FLAT
    - **大数据集(>10M)**: IVF_PQ 或 HNSW
    - **追求高召回**: HNSW
    - **节省内存**: IVF_SQ8 或 IVF_PQ

    ### 2. 优化搜索性能

    ```python
    # 调整nprobe参数（IVF索引）
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}

    # 调整ef参数（HNSW索引）
    search_params = {"metric_type": "L2", "params": {"ef": 64}}
    ```

    ### 3. 使用分区

    ```python
    # 按时间分区
    partition_2024 = collection.create_partition("2024")
    partition_2024.insert(data)

    # 只搜索特定分区
    collection.search(data, partition_names=["2024"])
    ```

    ### 4. 批量操作

    ```python
    # 批量插入（推荐每批1000-5000条）
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        collection.insert(batch)
    ```

    ### 5. 资源管理

    ```python
    # 使用完毕后释放内存
    collection.release()

    # 断开连接
    connections.disconnect('default')
    ```

    ## 🔗 学习资源

    - [Milvus官方文档](https://milvus.io/docs)
    - [PyMilvus API文档](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)
    - [Milvus GitHub](https://github.com/milvus-io/milvus)
    - [Milvus示例](https://github.com/milvus-io/pymilvus/tree/master/examples)
    """
    )
    return


if __name__ == "__main__":
    app.run()
