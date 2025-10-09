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
    # 使用Milvus构建RAG系统

    ![RAG Demo](https://raw.githubusercontent.com/milvus-io/bootcamp/master/tutorials/quickstart/apps/rag_search_with_milvus/pics/rag_demo.png)

    在本教程中，我们将展示如何使用Milvus构建RAG（检索增强生成）管道。

    RAG系统将检索系统与生成模型相结合，根据给定的提示生成新文本。系统首先使用Milvus从语料库中检索相关文档，然后使用生成模型基于检索到的文档生成新文本。

    ## 📦 准备工作

    ### 依赖和环境

    需要安装以下依赖：

    ```bash
    # 安装基础依赖
    uv pip install pymilvus openai requests tqdm python-dotenv
    ```

    ### Milvus服务器

    需要启动Milvus服务器（使用Docker）：

    ```bash
    # 下载docker-compose配置
    wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

    # 启动Milvus
    docker-compose up -d

    # 检查状态
    docker-compose ps
    ```

    Milvus默认端口：`19530`

    ### API密钥配置

    我们将使用OpenAI作为LLM。你需要准备[API密钥](https://platform.openai.com/docs/quickstart) `OPENAI_API_KEY`。

    在项目根目录创建`.env`文件：
    ```
    OPENAI_API_KEY=your_api_key_here
    OPENAI_API_BASE_URL=https://api.openai.com/v1
    ```
    """
    )
    return


@app.cell
def _():
    import os
    from dotenv import load_dotenv

    # 加载.env文件
    load_dotenv()

    print("=" * 60)
    print("🔑 加载API配置")
    print("=" * 60)

    # 获取配置
    openai_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    print(f"\n✅ 配置加载成功")
    print(f"📌 API Base URL: {openai_base_url}")
    print(f"🔑 API Key: {openai_api_key[:20]}..." if openai_api_key else "❌ API Key未设置")

    return load_dotenv, openai_api_key, openai_base_url, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣ 准备数据

    我们使用[Milvus文档2.4.x](https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip)的FAQ页面作为RAG中的私有知识，这是一个简单RAG管道的良好数据源。

    ### 下载和提取文档

    下载zip文件并将文档提取到文件夹`milvus_docs`。

    **注意**: 由于网络原因，这一步可能需要一些时间。如果下载失败，可以手动下载并解压。
    """
    )
    return


@app.cell
def _():
    import requests
    import zipfile
    from pathlib import Path

    print("=" * 60)
    print("📥 下载Milvus文档")
    print("=" * 60)

    # 文件URL和路径
    url = "https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip"
    zip_path = "milvus_docs_2.4.x_en.zip"
    extract_dir = "milvus_docs"

    # 检查是否已经下载和解压
    if Path(extract_dir).exists() and any(Path(extract_dir).iterdir()):
        print("\n✅ 文档已存在，跳过下载")
    else:
        try:
            # 下载文档
            print("\n正在下载文档...")
            print(f"URL: {url}")

            _response = requests.get(url, stream=True)
            _response.raise_for_status()

            total_size = int(_response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0

            with open(zip_path, 'wb') as f:
                for chunk in _response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # 显示进度
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r下载进度: {progress:.1f}%", end='')

            print("\n✅ 下载完成")

            # 解压文档
            print("\n正在解压文档...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print("✅ 解压完成")

            # 删除zip文件
            Path(zip_path).unlink()
            print("🗑️  已删除临时文件")

        except Exception as _e:
            print(f"\n❌ 下载或解压失败: {_e}")
            print("\n💡 解决方案:")
            print("   1. 检查网络连接")
            print("   2. 手动下载: https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip")
            print("   3. 手动解压到 milvus_docs 文件夹")

    return Path, requests, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 加载文档

    我们从文件夹`milvus_docs/en/faq`加载所有markdown文件。对于每个文档，我们简单地使用"# "来分隔文件中的内容，这可以粗略地分隔markdown文件每个主要部分的内容。
    """
    )
    return


@app.cell
def _():
    from glob import glob

    print("=" * 60)
    print("📚 加载文档")
    print("=" * 60)

    text_lines = []

    for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
        with open(file_path, "r", encoding="utf-8") as _file:
            file_text = _file.read()
        text_lines += file_text.split("# ")

    # 过滤空行
    text_lines = [_line.strip() for _line in text_lines if _line.strip()]

    print(f"\n✅ 加载完成")
    print(f"📊 文档片段数量: {len(text_lines)}")
    print(f"\n📝 示例片段（前200字符）:")
    print(text_lines[0][:200] + "...")

    return file_text, glob, text_lines


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2️⃣ 准备嵌入模型

    我们初始化OpenAI客户端来准备嵌入模型。

    我们使用[text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)模型作为示例。
    """
    )
    return


@app.cell
def _(openai_api_key, openai_base_url):
    from openai import OpenAI

    print("=" * 60)
    print("🤖 初始化OpenAI客户端")
    print("=" * 60)

    # 创建OpenAI客户端
    openai_client = OpenAI(
        base_url=openai_base_url,
        api_key=openai_api_key
    )

    # 定义嵌入函数
    def emb_text(text):
        """生成文本嵌入向量"""
        return (
            openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            .data[0]
            .embedding
        )

    print(f"\n✅ OpenAI客户端已初始化")
    print(f"📌 Base URL: {openai_base_url}")
    print(f"📌 使用模型: text-embedding-3-small")

    # 生成测试嵌入
    print("\n🧪 测试嵌入生成...")
    test_embedding = emb_text("This is a test")
    embedding_dim = len(test_embedding)

    print(f"✅ 测试成功")
    print(f"📏 嵌入维度: {embedding_dim}")
    print(f"📊 前10个元素: {test_embedding[:10]}")

    return OpenAI, emb_text, embedding_dim, openai_client, test_embedding


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3️⃣ 将数据加载到Milvus

    ### 连接到Milvus服务器

    我们使用标准版Milvus服务器来存储数据。

    **连接参数**:
    - `host`: Milvus服务器地址（默认：localhost）
    - `port`: Milvus服务器端口（默认：19530）
    - 如果你想使用[Zilliz Cloud](https://zilliz.com/cloud)（Milvus的完全托管云服务），请使用相应的连接参数
    """
    )
    return


@app.cell
def _(embedding_dim):
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

    print("=" * 60)
    print("🗄️  连接Milvus并创建Collection")
    print("=" * 60)

    try:
        # 连接到Milvus服务器
        print("\n正在连接到Milvus服务器...")
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
        print("✅ 成功连接到Milvus服务器")
        print(f"📌 Milvus版本: {utility.get_server_version()}")

        collection_name = "my_rag_collection"

        # 检查并删除已存在的集合
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"\n🗑️  已删除旧集合: {collection_name}")

        # 定义Collection的Schema
        print("\n正在创建Collection Schema...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields=fields, description="RAG文档集合")

        # 创建Collection
        print("正在创建Collection...")
        collection = Collection(name=collection_name, schema=schema)

        print(f"\n✅ 创建集合: {collection_name}")
        print(f"📏 向量维度: {embedding_dim}")
        print(f"📊 字段: id, vector, text")

    except Exception as _e:
        print(f"\n❌ Milvus连接或初始化失败: {_e}")
        print("\n💡 解决方案:")
        print("   1. 确保Milvus服务已启动: docker-compose ps")
        print("   2. 检查端口19530是否可访问")
        print("   3. 查看Milvus日志: docker-compose logs milvus-standalone")
        raise

    return Collection, CollectionSchema, DataType, FieldSchema, collection, collection_name, connections, utility


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 插入数据

    遍历文本行，创建嵌入，然后将数据插入Milvus。

    这里有一个新字段`text`，它是集合schema中未定义的字段。它将自动添加到保留的JSON动态字段中，可以在高层次上作为普通字段处理。
    """
    )
    return


@app.cell
def _(collection, emb_text, text_lines):
    from tqdm import tqdm

    print("=" * 60)
    print("📥 插入数据到Milvus")
    print("=" * 60)

    print(f"\n准备插入 {len(text_lines)} 条文档...")
    print("⏳ 正在生成嵌入向量（这可能需要几分钟）...")

    # 准备数据
    ids = []
    vectors = []
    texts = []

    for _i, _line in enumerate(tqdm(text_lines, desc="生成嵌入")):
        ids.append(_i)
        vectors.append(emb_text(_line))
        texts.append(_line)

    # 插入数据
    insert_result = collection.insert([ids, vectors, texts])

    print(f"\n✅ 插入成功")
    print(f"📊 插入数量: {insert_result.insert_count}")
    print(f"💾 数据已存储到Milvus")

    # 刷新数据（确保数据持久化）
    collection.flush()
    print("✅ 数据已刷新到磁盘")

    return ids, insert_result, texts, tqdm, vectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 创建索引并加载Collection

    在搜索之前，需要为向量字段创建索引并加载Collection到内存。
    """
    )
    return


@app.cell
def _(collection):
    print("=" * 60)
    print("🔧 创建索引")
    print("=" * 60)

    # 为向量字段创建索引
    print("\n正在创建IVF_FLAT索引...")
    index_params = {
        "metric_type": "IP",  # 内积距离
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index(
        field_name="vector",
        index_params=index_params
    )

    print("✅ 索引创建成功")
    print(f"📊 索引类型: IVF_FLAT")
    print(f"📏 距离度量: IP（内积）")

    # 加载Collection到内存
    print("\n正在加载Collection到内存...")
    collection.load()
    print("✅ Collection已加载")

    return (index_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4️⃣ 构建RAG

    ### 检索查询数据

    让我们指定一个关于Milvus的常见问题。
    """
    )
    return


@app.cell
def _():
    question = "How is data stored in milvus?"

    print("=" * 60)
    print("❓ 用户问题")
    print("=" * 60)
    print(f"\n问题: {question}")
    return (question,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 在集合中搜索问题

    在集合中搜索问题并检索语义上最相关的前3个匹配项。
    """
    )
    return


@app.cell
def _(collection, emb_text, question):
    print("=" * 60)
    print("🔍 向量搜索")
    print("=" * 60)

    print("\n正在搜索相关文档...")

    # 搜索参数
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }

    # 执行搜索
    search_res = collection.search(
        data=[emb_text(question)],  # 将问题转换为嵌入向量
        anns_field="vector",  # 向量字段名
        param=search_params,
        limit=3,  # 返回前3个结果
        output_fields=["text"]  # 返回text字段
    )

    print(f"✅ 搜索完成")
    print(f"📊 找到 {len(search_res[0])} 个相关文档")
    return search_params, search_res


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 查看搜索结果

    让我们看看查询的搜索结果。
    """
    )
    return


@app.cell
def _(search_res):
    import json

    print("=" * 60)
    print("📄 检索到的文档")
    print("=" * 60)

    # 检查搜索结果是否为空
    if not search_res or len(search_res) == 0 or len(search_res[0]) == 0:
        print("\n⚠️  未找到相关文档")
        retrieved_lines_with_distances = []
    else:
        retrieved_lines_with_distances = [
            (hit.entity.get("text"), hit.distance) for hit in search_res[0]
        ]

        print("\n相关文档及其相似度分数:\n")
        for _idx, (_text, _distance) in enumerate(retrieved_lines_with_distances, 1):
            print(f"文档 {_idx} (相似度: {_distance:.4f}):")
            print(f"{_text[:200]}...")
            print("-" * 60)

    return json, retrieved_lines_with_distances


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5️⃣ 使用LLM获取RAG响应

    ### 准备上下文

    将检索到的文档转换为字符串格式。
    """
    )
    return


@app.cell
def _(retrieved_lines_with_distances):
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    print("=" * 60)
    print("📝 构建上下文")
    print("=" * 60)
    print(f"\n上下文长度: {len(context)} 字符")
    print(f"\n上下文预览（前300字符）:")
    print(context[:300] + "...")
    return (context,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 定义提示词

    为语言模型定义系统和用户提示词。此提示词与从Milvus检索到的文档组装在一起。
    """
    )
    return


@app.cell
def _(context, question):
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """

    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    print("=" * 60)
    print("💬 提示词配置")
    print("=" * 60)
    print("\n系统提示词:")
    print(SYSTEM_PROMPT.strip())
    print("\n用户提示词（前200字符）:")
    print(USER_PROMPT[:200] + "...")
    return SYSTEM_PROMPT, USER_PROMPT


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 生成RAG响应

    使用OpenAI ChatGPT基于提示词生成响应。
    """
    )
    return


@app.cell
def _(SYSTEM_PROMPT, USER_PROMPT, openai_client):
    print("=" * 60)
    print("🤖 生成RAG响应")
    print("=" * 60)

    print("\n正在调用OpenAI API...")

    chat_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    rag_answer = chat_response.choices[0].message.content

    print("✅ 响应生成完成\n")
    print("=" * 60)
    print("💡 RAG系统回答")
    print("=" * 60)
    print(f"\n{rag_answer}\n")
    print("=" * 60)
    return rag_answer, chat_response


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 总结

    ### RAG系统工作流程

    1. **数据准备**: 加载Milvus文档并分割成片段
    2. **向量化**: 使用OpenAI嵌入模型将文本转换为向量
    3. **存储**: 将向量和文本存储到Milvus向量数据库
    4. **检索**: 根据用户问题检索最相关的文档片段
    5. **生成**: 使用LLM基于检索到的上下文生成答案

    ### 关键组件

    | 组件 | 技术 | 作用 |
    |------|------|------|
    | **向量数据库** | Milvus Lite | 存储和检索向量 |
    | **嵌入模型** | text-embedding-3-small | 文本向量化 |
    | **LLM** | GPT-3.5-turbo | 生成答案 |
    | **距离度量** | 内积（IP） | 计算相似度 |

    ### 优化建议

    1. **调整检索数量**: 修改`limit`参数来控制检索的文档数量
    2. **使用更好的分割策略**: 使用更智能的文档分割方法
    3. **优化提示词**: 改进系统和用户提示词以获得更好的答案
    4. **使用更强大的模型**: 尝试GPT-4或其他更强大的模型
    5. **添加重排序**: 在检索后添加重排序步骤提高相关性

    ## 🔗 相关资源

    - [Milvus官方文档](https://milvus.io/docs)
    - [OpenAI API文档](https://platform.openai.com/docs)
    - [RAG示例应用](https://github.com/milvus-io/bootcamp/tree/master/tutorials/quickstart/apps/rag_search_with_milvus)
    """
    )
    return


if __name__ == "__main__":
    app.run()
