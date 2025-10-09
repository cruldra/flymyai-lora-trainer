"""
Chroma 完全指南 - AI原生嵌入式向量数据库

Chroma是一个开源的嵌入式向量数据库，专为AI应用设计，
提供简单易用的API和强大的向量搜索功能。

特点：
1. 🚀 开箱即用 - 无需配置，直接使用
2. 🔍 向量搜索 - 支持多种距离度量
3. 🎯 元数据过滤 - 灵活的过滤查询
4. 📦 嵌入式 - 可以作为Python库直接使用
5. 🌐 多模态 - 支持文本、图像等多种数据类型

作者: Marimo Notebook
日期: 2025-01-XX
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="Chroma 完全指南")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🎨 Chroma 完全指南

    ## 什么是Chroma？

    **Chroma** 是一个AI原生的开源嵌入式向量数据库。它的设计理念是让开发者能够快速构建基于LLM的应用，无需复杂的配置和部署。

    ### 核心特性

    1. **开箱即用** 🚀
       - 零配置启动
       - 嵌入式数据库
       - 持久化存储

    2. **简单易用** 📝
       - 直观的Python API
       - 自动生成嵌入向量
       - 内置多种嵌入模型

    3. **强大搜索** 🔍
       - 语义相似度搜索
       - 元数据过滤
       - 混合查询

    4. **灵活部署** 🌐
       - 嵌入式模式（本地）
       - 客户端-服务器模式
       - Docker部署

    5. **生态集成** 🔗
       - LangChain集成
       - LlamaIndex集成
       - OpenAI兼容

    ### Chroma vs 其他向量数据库

    | 特性 | Chroma | Milvus | Qdrant | Pinecone |
    |------|--------|--------|--------|----------|
    | 部署复杂度 | ⭐ 极简 | ⭐⭐⭐ 复杂 | ⭐⭐ 中等 | ⭐ 托管 |
    | 性能 | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐⭐ 很好 | ⭐⭐⭐⭐ 很好 |
    | 开源 | ✅ | ✅ | ✅ | ❌ |
    | 嵌入式 | ✅ | ❌ | ✅ | ❌ |
    | 适用场景 | 原型/小型应用 | 大规模生产 | 中大型应用 | 托管服务 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📦 安装和部署

    ### 方式1: Python库安装（推荐用于开发）

    ```bash
    # 基础安装
    pip install chromadb

    # 或使用uv
    uv pip install chromadb
    ```

    ### 方式2: Docker部署（推荐用于生产）

    ```bash
    # 拉取镜像
    docker pull chromadb/chroma

    # 运行容器
    docker run -p 8000:8000 chromadb/chroma
    ```

    **访问地址：**
    - HTTP API: `http://localhost:8000`
    - 健康检查: `http://localhost:8000/api/v1/heartbeat`

    ### 方式3: Docker Compose

    ```yaml
    version: '3.8'
    services:
      chroma:
        image: chromadb/chroma:latest
        ports:
          - "8000:8000"
        volumes:
          - ./chroma_data:/chroma/chroma
        environment:
          - IS_PERSISTENT=TRUE
    ```

    ```bash
    docker-compose up -d
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣ 快速开始

    ### 创建客户端

    Chroma支持两种模式：
    - **嵌入式模式**：数据存储在本地
    - **客户端-服务器模式**：连接到远程Chroma服务器
    """
    )
    return


@app.cell
def _():
    import chromadb
    from chromadb.config import Settings

    print("=" * 60)
    print("🔌 创建Chroma客户端")
    print("=" * 60)

    # 方式1: 嵌入式客户端（数据存储在内存中）
    client_memory = chromadb.Client()
    print("\n✅ 创建内存客户端成功")
    print("   数据存储: 内存（临时）")

    # 方式2: 持久化客户端（数据存储在磁盘）
    client_persistent = chromadb.PersistentClient(path="./chroma_db")
    print("\n✅ 创建持久化客户端成功")
    print("   数据存储: ./chroma_db（持久化）")

    # 方式3: HTTP客户端（连接到远程服务器）
    # client_http = chromadb.HttpClient(host="localhost", port=8000)
    # print("\n✅ 创建HTTP客户端成功")
    # print("   服务器地址: http://localhost:8000")

    # 使用持久化客户端进行后续操作
    client = client_persistent

    print("\n📊 客户端信息:")
    print(f"   心跳检测: {client.heartbeat()}ms")
    return Settings, chromadb, client, client_memory, client_persistent


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2️⃣ Collection（集合）管理

    Collection是Chroma中存储向量的容器，类似于关系数据库中的表。

    ### Collection的核心概念

    - **名称**：唯一标识符
    - **嵌入函数**：将文本转换为向量的函数
    - **元数据**：附加信息（可选）
    - **距离度量**：计算向量相似度的方法
    """
    )
    return


@app.cell
def _(client):
    print("=" * 60)
    print("📚 Collection管理")
    print("=" * 60)

    # 创建或获取Collection
    my_collection = client.get_or_create_collection(
        name="my_collection",
        metadata={"description": "我的第一个Chroma集合"}
    )

    print("\n✅ Collection创建/获取成功")
    print(f"   名称: {my_collection.name}")
    print(f"   ID: {my_collection.id}")
    print(f"   元数据: {my_collection.metadata}")

    # 列出所有Collections
    all_collections = client.list_collections()
    print(f"\n📋 所有Collections ({len(all_collections)}个):")
    for list_coll in all_collections:
        print(f"   - {list_coll.name}")

    # 获取Collection信息
    collection_count = my_collection.count()
    print(f"\n📊 Collection统计:")
    print(f"   文档数量: {collection_count}")

    return my_collection, all_collections, collection_count


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3️⃣ 添加数据

    ### 数据结构

    Chroma中的每条数据包含：
    - **documents**：原始文本内容
    - **embeddings**：向量表示（可选，自动生成）
    - **metadatas**：元数据字典（可选）
    - **ids**：唯一标识符

    ### 添加方式

    | 方法 | 说明 | 使用场景 |
    |------|------|----------|
    | `add()` | 添加新数据 | 插入新文档 |
    | `upsert()` | 更新或插入 | 更新已存在的文档 |
    | `update()` | 更新数据 | 修改已存在的文档 |
    """
    )
    return


@app.cell
def _(my_collection):
    print("=" * 60)
    print("📥 添加数据到Collection")
    print("=" * 60)

    # 准备示例数据
    sample_documents = [
        "Chroma是一个开源的向量数据库",
        "向量数据库用于存储和检索嵌入向量",
        "机器学习模型可以将文本转换为向量",
        "语义搜索比关键词搜索更智能",
        "RAG系统结合了检索和生成能力",
        "LangChain是一个流行的LLM应用框架",
        "OpenAI提供强大的嵌入模型",
        "向量相似度可以用余弦距离计算",
        "HNSW是一种高效的向量索引算法",
        "嵌入向量捕获了文本的语义信息"
    ]

    sample_metadatas = [
        {"category": "database", "source": "docs", "page": 1},
        {"category": "database", "source": "docs", "page": 1},
        {"category": "ml", "source": "tutorial", "page": 2},
        {"category": "search", "source": "blog", "page": 1},
        {"category": "rag", "source": "docs", "page": 3},
        {"category": "framework", "source": "docs", "page": 1},
        {"category": "ml", "source": "api", "page": 1},
        {"category": "algorithm", "source": "paper", "page": 5},
        {"category": "algorithm", "source": "paper", "page": 8},
        {"category": "ml", "source": "tutorial", "page": 3}
    ]

    sample_ids = [f"doc_{doc_i}" for doc_i in range(len(sample_documents))]

    print(f"\n准备添加 {len(sample_documents)} 条文档...")

    # 添加数据（Chroma会自动生成嵌入向量）
    my_collection.add(
        documents=sample_documents,
        metadatas=sample_metadatas,
        ids=sample_ids
    )

    print(f"✅ 数据添加成功")
    print(f"📊 当前文档数量: {my_collection.count()}")

    # 显示部分数据
    print(f"\n📝 示例文档:")
    for sample_idx in range(3):
        print(f"   {sample_idx+1}. ID: {sample_ids[sample_idx]}")
        print(f"      文本: {sample_documents[sample_idx]}")
        print(f"      元数据: {sample_metadatas[sample_idx]}")
        print()

    return sample_documents, sample_ids, sample_metadatas


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4️⃣ 查询数据

    ### 查询类型

    | 类型 | 方法 | 说明 |
    |------|------|------|
    | 语义搜索 | `query()` | 基于文本相似度搜索 |
    | 精确查询 | `get()` | 根据ID或元数据获取 |
    | 混合查询 | `query()` + `where` | 语义搜索+元数据过滤 |

    ### 查询参数

    - `query_texts`: 查询文本列表
    - `query_embeddings`: 查询向量（可选）
    - `n_results`: 返回结果数量
    - `where`: 元数据过滤条件
    - `where_document`: 文档内容过滤
    - `include`: 返回的字段
    """
    )
    return


@app.cell
def _(my_collection):
    print("=" * 60)
    print("🔍 查询数据")
    print("=" * 60)

    # 1. 语义搜索
    print("\n1️⃣ 语义搜索:")
    print("   查询: '什么是向量数据库？'")

    query_results = my_collection.query(
        query_texts=["什么是向量数据库？"],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n   找到 {len(query_results['documents'][0])} 个相关文档:")
    for q_idx, (q_doc, q_meta, q_dist) in enumerate(zip(
        query_results['documents'][0],
        query_results['metadatas'][0],
        query_results['distances'][0]
    )):
        print(f"\n   {q_idx+1}. 文档: {q_doc}")
        print(f"      元数据: {q_meta}")
        print(f"      距离: {q_dist:.4f}")

    # 2. 带元数据过滤的查询
    print("\n" + "=" * 60)
    print("2️⃣ 元数据过滤查询:")
    print("   查询: '机器学习' + category='ml'")

    filtered_results = my_collection.query(
        query_texts=["机器学习"],
        n_results=3,
        where={"category": "ml"},
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n   找到 {len(filtered_results['documents'][0])} 个相关文档:")
    for f_idx, (f_doc, f_meta, f_dist) in enumerate(zip(
        filtered_results['documents'][0],
        filtered_results['metadatas'][0],
        filtered_results['distances'][0]
    )):
        print(f"\n   {f_idx+1}. 文档: {f_doc}")
        print(f"      元数据: {f_meta}")
        print(f"      距离: {f_dist:.4f}")

    return query_results, filtered_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5️⃣ 元数据过滤

    ### 过滤操作符

    Chroma支持丰富的元数据过滤操作：

    | 操作符 | 说明 | 示例 |
    |--------|------|------|
    | `$eq` | 等于 | `{"category": {"$eq": "ml"}}` |
    | `$ne` | 不等于 | `{"category": {"$ne": "ml"}}` |
    | `$gt` | 大于 | `{"page": {"$gt": 5}}` |
    | `$gte` | 大于等于 | `{"page": {"$gte": 5}}` |
    | `$lt` | 小于 | `{"page": {"$lt": 5}}` |
    | `$lte` | 小于等于 | `{"page": {"$lte": 5}}` |
    | `$in` | 在列表中 | `{"category": {"$in": ["ml", "rag"]}}` |
    | `$nin` | 不在列表中 | `{"category": {"$nin": ["ml", "rag"]}}` |
    | `$and` | 逻辑与 | `{"$and": [{"page": {"$gt": 1}}, {"category": "ml"}]}` |
    | `$or` | 逻辑或 | `{"$or": [{"category": "ml"}, {"category": "rag"}]}` |
    """
    )
    return


@app.cell
def _(my_collection):
    print("=" * 60)
    print("🎯 元数据过滤示例")
    print("=" * 60)

    # 1. 简单过滤
    print("\n1️⃣ 简单过滤 - category='ml':")
    filter_results1 = my_collection.get(
        where={"category": "ml"},
        include=["documents", "metadatas"]
    )
    print(f"   找到 {len(filter_results1['documents'])} 个文档")
    for fr1_doc in filter_results1['documents'][:2]:
        print(f"   - {fr1_doc}")

    # 2. 范围过滤
    print("\n2️⃣ 范围过滤 - page > 2:")
    filter_results2 = my_collection.get(
        where={"page": {"$gt": 2}},
        include=["documents", "metadatas"]
    )
    print(f"   找到 {len(filter_results2['documents'])} 个文档")
    for fr2_doc, fr2_meta in zip(filter_results2['documents'][:2], filter_results2['metadatas'][:2]):
        print(f"   - {fr2_doc} (page: {fr2_meta['page']})")

    # 3. 列表过滤
    print("\n3️⃣ 列表过滤 - category in ['ml', 'rag']:")
    filter_results3 = my_collection.get(
        where={"category": {"$in": ["ml", "rag"]}},
        include=["documents", "metadatas"]
    )
    print(f"   找到 {len(filter_results3['documents'])} 个文档")
    for fr3_doc, fr3_meta in zip(filter_results3['documents'][:2], filter_results3['metadatas'][:2]):
        print(f"   - {fr3_doc} (category: {fr3_meta['category']})")

    # 4. 复合过滤
    print("\n4️⃣ 复合过滤 - category='ml' AND page >= 2:")
    filter_results4 = my_collection.get(
        where={
            "$and": [
                {"category": "ml"},
                {"page": {"$gte": 2}}
            ]
        },
        include=["documents", "metadatas"]
    )
    print(f"   找到 {len(filter_results4['documents'])} 个文档")
    for fr4_doc, fr4_meta in zip(filter_results4['documents'], filter_results4['metadatas']):
        print(f"   - {fr4_doc} (category: {fr4_meta['category']}, page: {fr4_meta['page']})")

    return filter_results1, filter_results2, filter_results3, filter_results4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 6️⃣ 嵌入函数（Embedding Functions）

    ### 内置嵌入函数

    Chroma提供多种内置嵌入函数：

    | 函数 | 说明 | 使用场景 |
    |------|------|----------|
    | `DefaultEmbeddingFunction` | 默认函数（sentence-transformers） | 通用文本 |
    | `OpenAIEmbeddingFunction` | OpenAI嵌入 | 高质量嵌入 |
    | `CohereEmbeddingFunction` | Cohere嵌入 | 多语言支持 |
    | `HuggingFaceEmbeddingFunction` | HuggingFace模型 | 自定义模型 |
    | `SentenceTransformerEmbeddingFunction` | Sentence Transformers | 本地嵌入 |

    ### 自定义嵌入函数

    你也可以实现自己的嵌入函数，只需实现 `__call__` 方法。
    """
    )
    return


@app.cell
def _(chromadb):
    print("=" * 60)
    print("🔧 使用不同的嵌入函数")
    print("=" * 60)

    # 1. 默认嵌入函数
    print("\n1️⃣ 默认嵌入函数:")
    default_embedding_func = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    print(f"   类型: {type(default_embedding_func).__name__}")

    # 2. Sentence Transformer嵌入函数
    print("\n2️⃣ Sentence Transformer嵌入函数:")
    st_embedding_func = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    print(f"   模型: all-MiniLM-L6-v2")
    print(f"   维度: 384")

    # 3. OpenAI嵌入函数（需要API密钥）
    print("\n3️⃣ OpenAI嵌入函数:")
    print("   需要设置 OPENAI_API_KEY 环境变量")
    print("   示例代码:")
    print("   ```python")
    print("   openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(")
    print("       api_key='your-api-key',")
    print("       model_name='text-embedding-3-small'")
    print("   )")
    print("   ```")

    # 测试嵌入函数
    print("\n4️⃣ 测试嵌入函数:")
    embedding_test_text = ["这是一个测试文本"]
    test_embeddings = st_embedding_func(embedding_test_text)
    print(f"   输入文本: {embedding_test_text[0]}")
    print(f"   嵌入维度: {len(test_embeddings[0])}")
    print(f"   嵌入向量前5个值: {test_embeddings[0][:5]}")

    return default_embedding_func, st_embedding_func, embedding_test_text, test_embeddings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 7️⃣ 更新和删除数据

    ### 数据操作

    | 操作 | 方法 | 说明 |
    |------|------|------|
    | 更新 | `update()` | 更新已存在的文档 |
    | 插入或更新 | `upsert()` | 如果存在则更新，否则插入 |
    | 删除 | `delete()` | 根据ID或条件删除 |
    """
    )
    return


@app.cell
def _(my_collection):
    print("=" * 60)
    print("✏️ 更新和删除数据")
    print("=" * 60)

    # 1. 更新数据
    print("\n1️⃣ 更新数据:")
    print("   更新 doc_0 的文档内容和元数据")

    my_collection.update(
        ids=["doc_0"],
        documents=["Chroma是一个强大的AI原生向量数据库"],
        metadatas=[{"category": "database", "source": "docs", "page": 1, "updated": True}]
    )

    # 验证更新
    update_result = my_collection.get(ids=["doc_0"], include=["documents", "metadatas"])
    print(f"   ✅ 更新后的文档: {update_result['documents'][0]}")
    print(f"   ✅ 更新后的元数据: {update_result['metadatas'][0]}")

    # 2. Upsert操作
    print("\n2️⃣ Upsert操作:")
    print("   插入新文档 doc_new")

    my_collection.upsert(
        ids=["doc_new"],
        documents=["这是一个新插入的文档"],
        metadatas=[{"category": "test", "source": "manual", "page": 1}]
    )

    print(f"   ✅ 当前文档数量: {my_collection.count()}")

    # 3. 删除数据
    print("\n3️⃣ 删除数据:")
    print("   删除 doc_new")

    my_collection.delete(ids=["doc_new"])
    print(f"   ✅ 删除后文档数量: {my_collection.count()}")

    # 4. 条件删除
    print("\n4️⃣ 条件删除:")
    print("   删除 category='test' 的所有文档")

    my_collection.delete(where={"category": "test"})
    print(f"   ✅ 删除后文档数量: {my_collection.count()}")

    return update_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 8️⃣ 距离度量

    ### 支持的距离度量

    Chroma支持多种距离度量方法：

    | 度量 | 说明 | 取值范围 | 适用场景 |
    |------|------|----------|----------|
    | `cosine` | 余弦距离（默认） | [0, 2] | 文本相似度 |
    | `l2` | 欧氏距离 | [0, ∞) | 空间距离 |
    | `ip` | 内积（点积） | (-∞, ∞) | 推荐系统 |

    ### 距离与相似度

    - **余弦距离**: 距离越小，相似度越高
    - **欧氏距离**: 距离越小，相似度越高
    - **内积**: 值越大，相似度越高（需要归一化向量）
    """
    )
    return


@app.cell
def _(client):
    print("=" * 60)
    print("📏 距离度量示例")
    print("=" * 60)

    # 创建使用不同距离度量的集合
    print("\n创建使用不同距离度量的集合...")

    # 余弦距离（默认）
    cosine_collection = client.get_or_create_collection(
        name="collection_cosine",
        metadata={"hnsw:space": "cosine"}
    )
    print("✅ 余弦距离集合创建成功")

    # 欧氏距离
    l2_collection = client.get_or_create_collection(
        name="collection_l2",
        metadata={"hnsw:space": "l2"}
    )
    print("✅ 欧氏距离集合创建成功")

    # 内积
    ip_collection = client.get_or_create_collection(
        name="collection_ip",
        metadata={"hnsw:space": "ip"}
    )
    print("✅ 内积集合创建成功")

    # 添加测试数据
    distance_test_docs = [
        "人工智能正在改变世界",
        "机器学习是AI的核心技术",
        "深度学习推动了AI的发展"
    ]
    distance_test_ids = ["test_1", "test_2", "test_3"]

    for metric_coll, metric_name in [(cosine_collection, "余弦"), (l2_collection, "欧氏"), (ip_collection, "内积")]:
        metric_coll.add(documents=distance_test_docs, ids=distance_test_ids)
        print(f"   {metric_name}距离集合: 添加 {len(distance_test_docs)} 条文档")

    return cosine_collection, ip_collection, l2_collection, distance_test_docs, distance_test_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 9️⃣ 实战案例：构建简单的RAG系统

    ### RAG系统架构

    ```
    用户问题 → 向量化 → 检索相关文档 → 组合上下文 → LLM生成答案
    ```

    ### 实现步骤

    1. 准备知识库文档
    2. 创建Chroma集合并添加文档
    3. 接收用户问题
    4. 检索相关文档
    5. 构建提示词
    6. 调用LLM生成答案
    """
    )
    return


@app.cell
def _(client):
    print("=" * 60)
    print("🤖 构建简单的RAG系统")
    print("=" * 60)

    # 1. 准备知识库
    print("\n1️⃣ 准备知识库:")
    rag_knowledge_base = [
        "Chroma是一个开源的AI原生向量数据库，专为LLM应用设计。",
        "向量数据库通过存储嵌入向量来实现语义搜索功能。",
        "RAG（检索增强生成）结合了信息检索和文本生成两种技术。",
        "嵌入向量是文本的数值表示，捕获了语义信息。",
        "Chroma支持自动生成嵌入向量，无需手动处理。",
        "语义搜索比传统关键词搜索更智能，能理解查询意图。",
        "LangChain是一个流行的框架，用于构建LLM应用。",
        "向量相似度搜索使用余弦距离或欧氏距离来衡量相似性。"
    ]

    print(f"   知识库文档数量: {len(rag_knowledge_base)}")

    # 2. 创建RAG集合
    print("\n2️⃣ 创建RAG集合:")
    rag_demo_collection = client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"description": "RAG系统知识库"}
    )

    # 添加知识库文档
    rag_demo_collection.add(
        documents=rag_knowledge_base,
        ids=[f"kb_{kb_i}" for kb_i in range(len(rag_knowledge_base))],
        metadatas=[{"source": "knowledge_base", "index": kb_i} for kb_i in range(len(rag_knowledge_base))]
    )

    print(f"   ✅ 添加 {rag_demo_collection.count()} 条知识库文档")

    # 3. 模拟用户问题
    print("\n3️⃣ 用户问题:")
    rag_user_question = "什么是RAG系统？"
    print(f"   问题: {rag_user_question}")

    # 4. 检索相关文档
    print("\n4️⃣ 检索相关文档:")
    rag_search_results = rag_demo_collection.query(
        query_texts=[rag_user_question],
        n_results=3,
        include=["documents", "distances"]
    )

    print(f"   找到 {len(rag_search_results['documents'][0])} 个相关文档:")
    for rag_idx, (rag_doc, rag_dist) in enumerate(zip(
        rag_search_results['documents'][0],
        rag_search_results['distances'][0]
    )):
        print(f"\n   {rag_idx+1}. {rag_doc}")
        print(f"      相似度距离: {rag_dist:.4f}")

    # 5. 构建上下文
    print("\n5️⃣ 构建提示词:")
    rag_context = "\n".join(rag_search_results['documents'][0])
    rag_prompt = f"""基于以下上下文回答问题：

上下文：
{rag_context}

问题：{rag_user_question}

答案："""

    print(f"   提示词长度: {len(rag_prompt)} 字符")
    print(f"\n   提示词预览:")
    print(f"   {rag_prompt[:200]}...")

    print("\n6️⃣ 下一步:")
    print("   将提示词发送给LLM（如OpenAI GPT、Claude等）生成答案")

    return rag_context, rag_knowledge_base, rag_prompt, rag_demo_collection, rag_search_results, rag_user_question


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔟 高级特性

    ### 1. 批量操作

    ```python
    # 批量添加
    collection.add(
        documents=large_document_list,
        ids=large_id_list,
        metadatas=large_metadata_list
    )

    # 批量查询
    results = collection.query(
        query_texts=["query1", "query2", "query3"],
        n_results=5
    )
    ```

    ### 2. Collection管理

    ```python
    # 删除Collection
    client.delete_collection(name="my_collection")

    # 重置客户端（删除所有数据）
    client.reset()

    # 获取Collection
    collection = client.get_collection(name="my_collection")
    ```

    ### 3. 持久化配置

    ```python
    # 自定义持久化路径
    client = chromadb.PersistentClient(
        path="./my_custom_path",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    ```

    ### 4. 性能优化

    - **批量操作**: 使用批量添加而不是逐条添加
    - **合适的n_results**: 不要检索过多结果
    - **元数据索引**: 合理设计元数据结构
    - **嵌入缓存**: 缓存常用查询的嵌入向量
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 API速查表

    ### 客户端操作

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `Client()` | 创建内存客户端 | `client = chromadb.Client()` |
    | `PersistentClient()` | 创建持久化客户端 | `client = chromadb.PersistentClient(path="./db")` |
    | `HttpClient()` | 创建HTTP客户端 | `client = chromadb.HttpClient(host="localhost")` |
    | `heartbeat()` | 心跳检测 | `client.heartbeat()` |
    | `list_collections()` | 列出所有集合 | `client.list_collections()` |
    | `get_collection()` | 获取集合 | `client.get_collection(name="my_col")` |
    | `get_or_create_collection()` | 获取或创建集合 | `client.get_or_create_collection(name="my_col")` |
    | `delete_collection()` | 删除集合 | `client.delete_collection(name="my_col")` |
    | `reset()` | 重置所有数据 | `client.reset()` |

    ### Collection操作

    | 方法 | 说明 | 示例 |
    |------|------|------|
    | `add()` | 添加数据 | `collection.add(documents=[...], ids=[...])` |
    | `upsert()` | 更新或插入 | `collection.upsert(documents=[...], ids=[...])` |
    | `update()` | 更新数据 | `collection.update(ids=[...], documents=[...])` |
    | `get()` | 获取数据 | `collection.get(ids=[...])` |
    | `query()` | 查询数据 | `collection.query(query_texts=[...], n_results=5)` |
    | `delete()` | 删除数据 | `collection.delete(ids=[...])` |
    | `count()` | 统计数量 | `collection.count()` |
    | `peek()` | 查看前N条 | `collection.peek(limit=10)` |

    ### 查询参数

    | 参数 | 类型 | 说明 |
    |------|------|------|
    | `query_texts` | List[str] | 查询文本列表 |
    | `query_embeddings` | List[List[float]] | 查询向量列表 |
    | `n_results` | int | 返回结果数量 |
    | `where` | Dict | 元数据过滤条件 |
    | `where_document` | Dict | 文档内容过滤 |
    | `include` | List[str] | 返回字段：documents, metadatas, distances, embeddings |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 💡 最佳实践

    ### 1. 选择合适的部署模式

    - **开发/原型**: 使用嵌入式模式（`Client()` 或 `PersistentClient()`）
    - **生产环境**: 使用客户端-服务器模式（`HttpClient()` + Docker）
    - **小规模应用**: 持久化客户端足够
    - **大规模应用**: 考虑使用Milvus或Qdrant

    ### 2. 数据组织

    - 使用有意义的ID（如 `doc_123` 而不是随机UUID）
    - 设计合理的元数据结构
    - 避免在单个Collection中存储过多数据（建议<100万条）
    - 使用多个Collection来组织不同类型的数据

    ### 3. 查询优化

    - 合理设置 `n_results`（通常3-10个结果足够）
    - 使用元数据过滤减少搜索空间
    - 缓存常用查询结果
    - 批量查询而不是单个查询

    ### 4. 嵌入函数选择

    - **通用文本**: 使用默认的sentence-transformers
    - **高质量**: 使用OpenAI embeddings
    - **多语言**: 使用多语言模型（如paraphrase-multilingual）
    - **特定领域**: 使用领域特定的微调模型

    ### 5. 错误处理

    ```python
    try:
        results = collection.query(query_texts=["test"], n_results=5)
    except Exception as e:
        print(f"查询失败: {e}")
        # 处理错误
    ```

    ### 6. 监控和维护

    - 定期检查Collection大小
    - 监控查询性能
    - 定期备份持久化数据
    - 清理不再使用的Collection
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔗 与其他工具集成

    ### LangChain集成

    ```python
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    # 创建向量存储
    vectorstore = Chroma(
        collection_name="my_collection",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )

    # 添加文档
    vectorstore.add_texts(texts=["doc1", "doc2"])

    # 相似度搜索
    results = vectorstore.similarity_search("query", k=3)
    ```

    ### LlamaIndex集成

    ```python
    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.vector_stores import ChromaVectorStore
    from llama_index.storage.storage_context import StorageContext

    # 创建向量存储
    chroma_collection = client.get_or_create_collection("my_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 创建索引
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    ```

    ### OpenAI集成

    ```python
    import openai
    from chromadb.utils import embedding_functions

    # 使用OpenAI嵌入函数
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key="your-api-key",
        model_name="text-embedding-3-small"
    )

    collection = client.create_collection(
        name="openai_collection",
        embedding_function=openai_ef
    )
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎓 总结

    ### Chroma的优势

    - ✅ **简单易用** - 零配置，开箱即用
    - ✅ **灵活部署** - 支持嵌入式和客户端-服务器模式
    - ✅ **功能完整** - 支持元数据过滤、多种嵌入函数
    - ✅ **生态友好** - 与LangChain、LlamaIndex无缝集成
    - ✅ **开源免费** - Apache 2.0许可证

    ### 适用场景

    - 🎯 **原型开发** - 快速验证想法
    - 📚 **知识库搜索** - 构建企业知识库
    - 🤖 **RAG应用** - 检索增强生成系统
    - 💬 **聊天机器人** - 基于上下文的对话
    - 🔍 **语义搜索** - 智能文档检索

    ### 何时不使用Chroma

    - ❌ **超大规模数据**（>1000万向量）→ 使用Milvus
    - ❌ **需要极致性能** → 使用Qdrant或Milvus
    - ❌ **复杂的分布式部署** → 使用Milvus
    - ❌ **需要托管服务** → 使用Pinecone

    ### 学习资源

    - 📖 [官方文档](https://docs.trychroma.com/)
    - 💻 [GitHub仓库](https://github.com/chroma-core/chroma)
    - 🎥 [视频教程](https://www.youtube.com/c/ChromaDB)
    - 💬 [Discord社区](https://discord.gg/MMeYNTmh3x)

    ---

    **恭喜！** 🎉 你已经掌握了Chroma的核心概念和API使用方法。

    现在你可以开始构建自己的AI应用了！
    """
    )
    return


if __name__ == "__main__":
    app.run()

