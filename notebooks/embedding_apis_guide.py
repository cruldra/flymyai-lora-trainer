"""
嵌入模型API完全指南

本笔记介绍主流厂商的嵌入模型API，包括：
- OpenAI Embeddings
- 阿里云DashScope
- Jina AI Embeddings
- Cohere Embeddings
- Voyage AI
- 与Sentence Transformers的对比

作者: Marimo Notebook
日期: 2025-01-XX
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="嵌入模型API完全指南")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🌐 嵌入模型API完全指南

    ## 什么是嵌入模型API？

    **嵌入模型API** 是云服务商提供的文本向量化服务，通过HTTP API调用即可获得高质量的文本嵌入向量。

    ### 🎯 主要厂商

    1. **OpenAI** - text-embedding-3系列
    2. **阿里云** - DashScope文本向量服务
    3. **Jina AI** - jina-embeddings系列
    4. **Cohere** - embed-multilingual系列
    5. **Voyage AI** - voyage-2系列

    ### ✨ 优势

    - 🚀 **无需部署** - 直接调用API
    - 🎯 **高质量** - 大规模数据训练
    - 📊 **可扩展** - 按需付费
    - 🔄 **持续更新** - 模型自动升级
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📦 安装依赖

    ```bash
    # OpenAI
    pip install openai

    # 阿里云DashScope
    pip install dashscope

    # Jina AI
    pip install requests  # 使用HTTP API

    # Cohere
    pip install cohere

    # 或使用uv
    uv pip install openai dashscope cohere
    ```
    """
    )
    return


@app.cell
def _():
    # 📦 导入必要的库
    import os
    import numpy as np
    from typing import List
    from dotenv import load_dotenv

    # 加载.env文件
    load_dotenv()

    print("=" * 60)
    print("🌐 嵌入模型API导入成功")
    print("=" * 60)
    print("\n🔑 API密钥加载状态:")

    # 检查密钥状态
    keys_status = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_API_BASE_URL": os.getenv("OPENAI_API_BASE_URL", ""),
        "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY", ""),
        "JINA_API_KEY": os.getenv("JINA_API_KEY", ""),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY", "")
    }

    for key_name, key_value in keys_status.items():
        if "BASE_URL" in key_name:
            # 显示完整URL
            if key_value:
                print(f"   ✅ {key_name}: {key_value}")
            else:
                print(f"   ℹ️  {key_name}: 使用默认 (https://api.openai.com/v1)")
        elif key_value and len(key_value) > 10:
            # 显示前缀和长度，隐藏实际密钥
            prefix = key_value[:7] if key_value.startswith("sk-") or key_value.startswith("jina_") else key_value[:4]
            print(f"   ✅ {key_name}: {prefix}...({len(key_value)}字符)")
        else:
            print(f"   ❌ {key_name}: 未配置")

    print("\n💡 提示: 在项目根目录创建 .env 文件配置密钥")
    print("   参考 .env.example 文件")
    return np, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔑 API密钥配置

    本笔记本使用 `.env` 文件管理API密钥，更安全便捷。

    ### 配置步骤

    1️⃣ **复制示例文件**
    ```bash
    cp .env.example .env
    ```

    2️⃣ **编辑 .env 文件**
    ```bash
    # .env 文件内容
    OPENAI_API_KEY=sk-your-actual-key
    OPENAI_API_BASE_URL=https://api.openai.com/v1  # 可选，自定义API地址
    DASHSCOPE_API_KEY=sk-your-actual-key
    JINA_API_KEY=jina_your-actual-key
    COHERE_API_KEY=your-actual-key
    ```

    ### 使用OpenRouter或其他兼容服务

    如果你想使用OpenRouter或其他OpenAI兼容的服务：

    ```bash
    # OpenRouter
    OPENAI_API_KEY=sk-or-v1-your-openrouter-key
    OPENAI_API_BASE_URL=https://openrouter.ai/api/v1

    # 其他兼容服务
    OPENAI_API_KEY=your-key
    OPENAI_API_BASE_URL=https://your-service.com/v1
    ```

    3️⃣ **重启笔记本**
    ```bash
    # 重新运行笔记本以加载新配置
    uv run marimo edit notebooks/embedding_apis_guide.py
    ```

    ### 获取API密钥

    - 🔵 **OpenAI**: https://platform.openai.com/api-keys
    - 🟠 **阿里云**: https://dashscope.console.aliyun.com/apiKey
    - 🟣 **Jina AI**: https://jina.ai/embeddings/
    - 🟢 **Cohere**: https://dashboard.cohere.com/api-keys

    ### 安全提示

    ⚠️ **重要**:
    - `.env` 文件已添加到 `.gitignore`，不会被提交到Git
    - 不要在代码中硬编码API密钥
    - 不要分享你的 `.env` 文件
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣ OpenAI Embeddings

    OpenAI提供业界领先的嵌入模型，支持多种维度和语言。

    ### 模型列表
    - `text-embedding-3-small` - 512/1536维，性价比高
    - `text-embedding-3-large` - 256/1024/3072维，最高质量
    - `text-embedding-ada-002` - 1536维，经典模型（已过时）
    """
    )
    return


@app.cell
def _(np, os):
    # 🔵 OpenAI Embeddings示例
    print("=" * 60)
    print("🔵 OpenAI Embeddings")
    print("=" * 60)

    # 检查API密钥
    openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_API_BASE_URL", "")

    if openai_key and openai_key.startswith("sk-"):
        try:
            from openai import OpenAI

            # 初始化客户端，支持自定义base_url
            client_kwargs = {"api_key": openai_key}
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url
                print(f"🔗 使用自定义API地址: {openai_base_url}")

            openai_client = OpenAI(**client_kwargs)

            # 示例文本
            openai_texts = [
                "人工智能正在改变世界",
                "机器学习是AI的核心技术",
                "今天天气真不错"
            ]

            # 调用API
            openai_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=openai_texts,
                encoding_format="float"
            )

            # 提取向量
            openai_embeddings = np.array([item.embedding for item in openai_response.data])

            print(f"✅ OpenAI嵌入成功")
            print(f"📊 模型: text-embedding-3-small")
            print(f"📏 向量维度: {openai_embeddings.shape}")
            print(f"💰 Token使用: {openai_response.usage.total_tokens}")
            print(f"\n前3个向量的前5维:")
            for openai_idx, openai_emb in enumerate(openai_embeddings[:3]):
                print(f"  {openai_idx+1}. {openai_emb[:5]}")

        except Exception as openai_error:
            print(f"❌ OpenAI调用失败: {openai_error}")
            openai_embeddings = None
    else:
        print("⚠️  未配置OPENAI_API_KEY，跳过OpenAI示例")
        print("   设置方法: export OPENAI_API_KEY='sk-...'")
        openai_embeddings = None

    return (openai_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2️⃣ 阿里云DashScope

    阿里云提供的文本向量服务，针对中文优化。

    ### 模型列表
    - `text-embedding-v1` - 1536维，通用文本
    - `text-embedding-v2` - 1536维，改进版
    - `text-embedding-v3` - 1024维，最新版本
    """
    )
    return


@app.cell
def _(np, os):
    # 🟠 阿里云DashScope示例
    print("=" * 60)
    print("🟠 阿里云DashScope Embeddings")
    print("=" * 60)

    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")

    if dashscope_key and dashscope_key.startswith("sk-"):
        try:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = dashscope_key

            # 示例文本
            dashscope_texts = [
                "深度学习需要大量数据",
                "神经网络模拟人脑结构",
                "我喜欢吃火锅"
            ]

            # 调用API
            dashscope_response = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v2,
                input=dashscope_texts
            )

            if dashscope_response.status_code == 200:
                # 提取向量
                dashscope_embeddings = np.array([
                    item['embedding'] for item in dashscope_response.output['embeddings']
                ])

                print(f"✅ DashScope嵌入成功")
                print(f"📊 模型: text-embedding-v2")
                print(f"📏 向量维度: {dashscope_embeddings.shape}")
                print(f"💰 Token使用: {dashscope_response.usage['total_tokens']}")
                print(f"\n前3个向量的前5维:")
                for dash_idx, dash_emb in enumerate(dashscope_embeddings[:3]):
                    print(f"  {dash_idx+1}. {dash_emb[:5]}")
            else:
                print(f"❌ DashScope调用失败: {dashscope_response.message}")
                dashscope_embeddings = None

        except Exception as dash_error:
            print(f"❌ DashScope调用失败: {dash_error}")
            dashscope_embeddings = None
    else:
        print("⚠️  未配置DASHSCOPE_API_KEY，跳过阿里云示例")
        print("   设置方法: export DASHSCOPE_API_KEY='sk-...'")
        dashscope_embeddings = None

    return (dashscope_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3️⃣ Jina AI Embeddings

    Jina AI提供开源和API两种方式，支持超长文本（8192 tokens）。

    ### 模型列表
    - `jina-embeddings-v2-base-en` - 768维，英文
    - `jina-embeddings-v2-base-zh` - 768维，中文
    - `jina-embeddings-v3` - 1024维，多语言
    """
    )
    return


@app.cell
def _(np, os):
    # 🟣 Jina AI Embeddings示例
    import requests

    print("=" * 60)
    print("🟣 Jina AI Embeddings")
    print("=" * 60)

    jina_key = os.getenv("JINA_API_KEY", "")

    if jina_key and jina_key.startswith("jina_"):
        try:
            # 示例文本
            jina_texts = [
                "自然语言处理是AI的重要分支",
                "计算机视觉让机器理解图像",
                "我在学习编程"
            ]

            # 调用API
            jina_url = "https://api.jina.ai/v1/embeddings"
            jina_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jina_key}"
            }
            jina_data = {
                "model": "jina-embeddings-v3",
                "input": jina_texts
            }

            jina_response = requests.post(jina_url, headers=jina_headers, json=jina_data)

            if jina_response.status_code == 200:
                jina_result = jina_response.json()
                jina_embeddings = np.array([item['embedding'] for item in jina_result['data']])

                print(f"✅ Jina AI嵌入成功")
                print(f"📊 模型: jina-embeddings-v3")
                print(f"📏 向量维度: {jina_embeddings.shape}")
                print(f"💰 Token使用: {jina_result['usage']['total_tokens']}")
                print(f"\n前3个向量的前5维:")
                for jina_idx, jina_emb in enumerate(jina_embeddings[:3]):
                    print(f"  {jina_idx+1}. {jina_emb[:5]}")
            else:
                print(f"❌ Jina AI调用失败: {jina_response.text}")
                jina_embeddings = None

        except Exception as jina_error:
            print(f"❌ Jina AI调用失败: {jina_error}")
            jina_embeddings = None
    else:
        print("⚠️  未配置JINA_API_KEY，跳过Jina AI示例")
        print("   设置方法: export JINA_API_KEY='jina_...'")
        jina_embeddings = None

    return (jina_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4️⃣ Cohere Embeddings

    Cohere提供多语言嵌入模型，支持100+种语言。

    ### 模型列表
    - `embed-english-v3.0` - 1024维，英文
    - `embed-multilingual-v3.0` - 1024维，多语言
    - `embed-english-light-v3.0` - 384维，轻量级
    """
    )
    return


@app.cell
def _(np, os):
    # 🟢 Cohere Embeddings示例
    print("=" * 60)
    print("🟢 Cohere Embeddings")
    print("=" * 60)

    cohere_key = os.getenv("COHERE_API_KEY", "")

    if cohere_key:
        try:
            import cohere

            cohere_client = cohere.Client(cohere_key)

            # 示例文本
            cohere_texts = [
                "强化学习通过奖励机制学习",
                "数据科学结合统计和编程",
                "我喜欢旅行"
            ]

            # 调用API
            cohere_response = cohere_client.embed(
                texts=cohere_texts,
                model="embed-multilingual-v3.0",
                input_type="search_document"
            )

            cohere_embeddings = np.array(cohere_response.embeddings)

            print(f"✅ Cohere嵌入成功")
            print(f"📊 模型: embed-multilingual-v3.0")
            print(f"📏 向量维度: {cohere_embeddings.shape}")
            print(f"\n前3个向量的前5维:")
            for cohere_idx, cohere_emb in enumerate(cohere_embeddings[:3]):
                print(f"  {cohere_idx+1}. {cohere_emb[:5]}")

        except Exception as cohere_error:
            print(f"❌ Cohere调用失败: {cohere_error}")
            cohere_embeddings = None
    else:
        print("⚠️  未配置COHERE_API_KEY，跳过Cohere示例")
        print("   设置方法: export COHERE_API_KEY='...'")
        cohere_embeddings = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📊 API对比：相似度计算

    使用不同API计算文本相似度，对比结果。
    """
    )
    return


@app.cell
def _(dashscope_embeddings, jina_embeddings, np, openai_embeddings):
    # 📊 相似度对比
    print("=" * 60)
    print("📊 不同API的相似度对比")
    print("=" * 60)

    def cosine_similarity(vec1, vec2):
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 测试句子对
    test_pairs = [
        ("句子1和句子2", 0, 1),  # 相关
        ("句子1和句子3", 0, 2),  # 不相关
    ]

    print("\n相似度对比 (值越接近1越相似):\n")
    print(f"{'API':<20} {'句子1-2':<12} {'句子1-3':<12}")
    print("-" * 50)

    # OpenAI
    if openai_embeddings is not None:
        sim_12 = cosine_similarity(openai_embeddings[0], openai_embeddings[1])
        sim_13 = cosine_similarity(openai_embeddings[0], openai_embeddings[2])
        print(f"{'OpenAI':<20} {sim_12:<12.4f} {sim_13:<12.4f}")

    # DashScope
    if dashscope_embeddings is not None:
        sim_12 = cosine_similarity(dashscope_embeddings[0], dashscope_embeddings[1])
        sim_13 = cosine_similarity(dashscope_embeddings[0], dashscope_embeddings[2])
        print(f"{'DashScope':<20} {sim_12:<12.4f} {sim_13:<12.4f}")

    # Jina AI
    if jina_embeddings is not None:
        sim_12 = cosine_similarity(jina_embeddings[0], jina_embeddings[1])
        sim_13 = cosine_similarity(jina_embeddings[0], jina_embeddings[2])
        print(f"{'Jina AI':<20} {sim_12:<12.4f} {sim_13:<12.4f}")

    print("\n💡 观察: 所有模型都能正确识别相关和不相关的句子对")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🆚 与Sentence Transformers的对比

    ### 架构对比

    | 维度 | API服务 | Sentence Transformers |
    |------|---------|----------------------|
    | **部署方式** | ☁️ 云端API | 💻 本地部署 |
    | **网络依赖** | ✅ 需要网络 | ❌ 无需网络 |
    | **成本模式** | 💰 按使用付费 | 🆓 一次性成本 |
    | **模型更新** | 🔄 自动更新 | 🔧 手动更新 |
    | **数据隐私** | ⚠️ 数据上传云端 | ✅ 数据本地处理 |
    | **性能** | 🚀 无需本地GPU | 🐌 依赖本地硬件 |
    | **可定制性** | ❌ 有限 | ✅ 完全可控 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎯 使用场景对比

    ### 适合使用API服务的场景

    ✅ **推荐使用API**:
    - 🚀 快速原型开发
    - 📊 小规模应用（<100万次/月）
    - 💼 企业级应用（需要SLA保障）
    - 🔄 需要最新模型
    - 💻 本地资源有限

    ### 适合使用Sentence Transformers的场景

    ✅ **推荐使用本地**:
    - 🔒 数据隐私要求高
    - 📈 大规模应用（>100万次/月）
    - ⚡ 低延迟要求
    - 🌐 离线环境
    - 🎨 需要模型微调
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 💰 成本对比

    ### API服务定价（参考）

    | 服务商 | 模型 | 价格 | 备注 |
    |--------|------|------|------|
    | **OpenAI** | text-embedding-3-small | $0.02/1M tokens | 性价比高 |
    | **OpenAI** | text-embedding-3-large | $0.13/1M tokens | 最高质量 |
    | **阿里云** | text-embedding-v2 | ¥0.0007/1K tokens | 中文优化 |
    | **Jina AI** | jina-embeddings-v3 | $0.02/1M tokens | 免费额度 |
    | **Cohere** | embed-multilingual-v3.0 | $0.10/1M tokens | 多语言 |

    ### Sentence Transformers成本

    | 项目 | 成本 | 备注 |
    |------|------|------|
    | **模型下载** | 免费 | 一次性 |
    | **GPU服务器** | $0.5-2/小时 | 云端GPU |
    | **本地GPU** | 一次性投入 | RTX 3060约$300 |
    | **运行成本** | 电费 | 可忽略 |

    ### 成本临界点

    假设每天处理100万个句子（约1亿tokens/月）：

    - **API成本**: $2-13/月
    - **本地成本**: GPU折旧 + 电费 ≈ $10-20/月

    💡 **结论**:
    - 小规模（<1000万tokens/月）→ 使用API
    - 大规模（>1亿tokens/月）→ 使用本地模型
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## ⚡ 性能对比

    ### 延迟对比（单次请求）

    | 方案 | 延迟 | 说明 |
    |------|------|------|
    | **OpenAI API** | 100-300ms | 网络 + 推理 |
    | **阿里云API** | 50-200ms | 国内网络快 |
    | **Jina AI API** | 100-300ms | 全球CDN |
    | **本地CPU** | 50-500ms | 取决于硬件 |
    | **本地GPU** | 10-50ms | 最快 |

    ### 吞吐量对比（批量处理）

    | 方案 | 吞吐量 | 说明 |
    |------|--------|------|
    | **API服务** | 1000-5000句/秒 | 受限于API限流 |
    | **本地GPU** | 5000-20000句/秒 | 取决于GPU性能 |
    | **本地CPU** | 100-500句/秒 | 较慢 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔧 实战：混合使用策略

    在实际应用中，可以结合API和本地模型的优势。
    """
    )
    return


@app.cell
def _():
    # 🔧 混合使用示例
    print("=" * 60)
    print("🔧 混合使用策略示例")
    print("=" * 60)

    class HybridEmbedding:
        """混合嵌入服务：优先使用本地，失败时回退到API"""

        def __init__(self, use_local=True, use_api_fallback=True):
            self.use_local = use_local
            self.use_api_fallback = use_api_fallback
            self.local_model = None

            if use_local:
                try:
                    from sentence_transformers import SentenceTransformer
                    self.local_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    print("✅ 本地模型加载成功")
                except Exception as local_err:
                    print(f"⚠️  本地模型加载失败: {local_err}")

        def embed(self, texts):
            """嵌入文本"""
            # 1. 尝试本地模型
            if self.local_model is not None:
                try:
                    return self.local_model.encode(texts)
                except Exception as local_err:
                    print(f"⚠️  本地推理失败: {local_err}")

            # 2. 回退到API
            if self.use_api_fallback:
                print("🔄 回退到API服务...")
                # 这里可以调用OpenAI或其他API
                return None

            return None

    # 使用示例
    hybrid_service = HybridEmbedding(use_local=True, use_api_fallback=True)

    print("\n💡 混合策略优势:")
    print("   1. 正常情况使用本地模型（快速、免费）")
    print("   2. 本地失败时自动切换到API（高可用）")
    print("   3. 可根据负载动态选择")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 主要API参考

    ### OpenAI Embeddings API

    ```python
    from openai import OpenAI

    # 标准用法
    client = OpenAI(api_key="sk-...")

    # 使用自定义API地址（如OpenRouter）
    client = OpenAI(
        api_key="sk-or-v1-...",
        base_url="https://openrouter.ai/api/v1"
    )

    response = client.embeddings.create(
        model="text-embedding-3-small",  # 或 text-embedding-3-large
        input=["文本1", "文本2"],
        encoding_format="float",  # 或 "base64"
        dimensions=512  # 可选，仅3-small/large支持
    )

    embeddings = [item.embedding for item in response.data]
    ```

    **参数说明**:
    - `api_key`: API密钥
    - `base_url`: API地址（可选，默认为OpenAI官方）
    - `model`: 模型名称
    - `input`: 字符串或字符串列表
    - `encoding_format`: 返回格式（float或base64）
    - `dimensions`: 输出维度（可选，用于降维）

    **支持的服务**:
    - OpenAI官方: `https://api.openai.com/v1`
    - OpenRouter: `https://openrouter.ai/api/v1`
    - Azure OpenAI: `https://{resource}.openai.azure.com/`
    - 其他兼容服务

    ---

    ### 阿里云DashScope API

    ```python
    import dashscope
    from dashscope import TextEmbedding

    dashscope.api_key = "sk-..."

    response = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v2,
        input=["文本1", "文本2"]
    )

    embeddings = [item['embedding'] for item in response.output['embeddings']]
    ```

    **参数说明**:
    - `model`: 模型名称（text_embedding_v1/v2/v3）
    - `input`: 字符串或字符串列表

    ---

    ### Jina AI Embeddings API

    ```python
    import requests

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer jina_..."
    }
    data = {
        "model": "jina-embeddings-v3",
        "input": ["文本1", "文本2"],
        "encoding_format": "float"
    }

    response = requests.post(url, headers=headers, json=data)
    embeddings = [item['embedding'] for item in response.json()['data']]
    ```

    **参数说明**:
    - `model`: 模型名称
    - `input`: 字符串或字符串列表
    - `encoding_format`: 返回格式

    ---

    ### Cohere Embeddings API

    ```python
    import cohere

    client = cohere.Client("...")

    response = client.embed(
        texts=["文本1", "文本2"],
        model="embed-multilingual-v3.0",
        input_type="search_document"  # 或 "search_query", "classification"
    )

    embeddings = response.embeddings
    ```

    **参数说明**:
    - `texts`: 字符串列表
    - `model`: 模型名称
    - `input_type`: 输入类型（影响向量优化方向）
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📊 完整对比表格

    ### 功能对比

    | 特性 | OpenAI | 阿里云 | Jina AI | Cohere | Sentence Transformers |
    |------|--------|--------|---------|--------|----------------------|
    | **部署方式** | API | API | API | API | 本地 |
    | **中文支持** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
    | **多语言** | ✅ | ✅ | ✅ | ✅ | ✅ |
    | **最大长度** | 8191 | 2048 | 8192 | 512 | 512 |
    | **向量维度** | 可调 | 固定 | 固定 | 固定 | 可选 |
    | **批量处理** | ✅ | ✅ | ✅ | ✅ | ✅ |
    | **免费额度** | ❌ | ❌ | ✅ | ✅ | ✅ |
    | **数据隐私** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ |
    | **离线使用** | ❌ | ❌ | ❌ | ❌ | ✅ |
    | **模型微调** | ❌ | ❌ | ❌ | ❌ | ✅ |

    ### 性能对比

    | 指标 | OpenAI | 阿里云 | Jina AI | Cohere | 本地GPU | 本地CPU |
    |------|--------|--------|---------|--------|---------|---------|
    | **延迟** | 100-300ms | 50-200ms | 100-300ms | 100-300ms | 10-50ms | 50-500ms |
    | **吞吐量** | 高 | 高 | 高 | 中 | 极高 | 低 |
    | **稳定性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

    ### 成本对比（每100万tokens）

    | 服务 | 成本 | 备注 |
    |------|------|------|
    | **OpenAI (small)** | $0.02 | 性价比最高 |
    | **OpenAI (large)** | $0.13 | 质量最高 |
    | **阿里云** | ¥0.70 (~$0.10) | 中文优化 |
    | **Jina AI** | $0.02 | 有免费额度 |
    | **Cohere** | $0.10 | 多语言强 |
    | **本地部署** | ~$0.01 | 电费+折旧 |

    ### 适用场景

    | 场景 | 推荐方案 | 原因 |
    |------|---------|------|
    | **快速原型** | OpenAI/Jina | 开箱即用 |
    | **中文应用** | 阿里云 | 中文优化 |
    | **大规模生产** | 本地部署 | 成本低 |
    | **数据敏感** | 本地部署 | 隐私保护 |
    | **多语言** | Cohere | 100+语言 |
    | **长文本** | Jina AI | 8192 tokens |
    | **低延迟** | 本地GPU | <50ms |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 💡 最佳实践建议

    ### 1. 选择合适的方案

    ```
    开始新项目
        ↓
    数据是否敏感？
        ├─ 是 → 使用本地模型
        └─ 否 → 继续
            ↓
    预算是否充足？
        ├─ 是 → 使用API（快速上线）
        └─ 否 → 使用本地模型
            ↓
    规模是否大？
        ├─ 是（>1亿tokens/月）→ 本地部署
        └─ 否 → API服务
    ```

    ### 2. API使用技巧

    ✅ **批量处理**
    ```python
    # ❌ 不好：逐个处理
    for text in texts:
        embedding = client.embed(text)

    # ✅ 好：批量处理
    embeddings = client.embed(texts)
    ```

    ✅ **错误处理**
    ```python
    import time

    def embed_with_retry(texts, max_retries=3):
        for attempt in range(max_retries):
            try:
                return client.embed(texts)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise
    ```

    ✅ **缓存结果**
    ```python
    import hashlib
    import json

    cache = {}

    def embed_with_cache(text):
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in cache:
            cache[key] = client.embed(text)
        return cache[key]
    ```

    ### 3. 本地部署技巧

    ✅ **模型选择**
    - 英文：`all-mpnet-base-v2`
    - 中文：`paraphrase-multilingual-mpnet-base-v2`
    - 速度优先：`all-MiniLM-L6-v2`

    ✅ **GPU加速**
    ```python
    model = SentenceTransformer('model-name', device='cuda')
    ```

    ✅ **批量优化**
    ```python
    embeddings = model.encode(
        texts,
        batch_size=64,  # 增大批量
        show_progress_bar=True
    )
    ```

    ### 4. 混合策略

    ```python
    class SmartEmbedding:
        def __init__(self):
            self.local_model = SentenceTransformer('...')
            self.api_client = OpenAI()
            self.cache = {}

        def embed(self, texts, use_cache=True):
            # 1. 检查缓存
            if use_cache:
                cached = self._get_cached(texts)
                if cached:
                    return cached

            # 2. 小批量用本地
            if len(texts) < 100:
                result = self.local_model.encode(texts)
            # 3. 大批量用API
            else:
                result = self.api_client.embed(texts)

            # 4. 缓存结果
            if use_cache:
                self._cache_result(texts, result)

            return result
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔗 资源链接

    ### OpenAI
    - 📖 [官方文档](https://platform.openai.com/docs/guides/embeddings)
    - 💰 [定价](https://openai.com/pricing)
    - 🔑 [获取API Key](https://platform.openai.com/api-keys)

    ### 阿里云DashScope
    - 📖 [官方文档](https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-api)
    - 💰 [定价](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-metering-and-billing)
    - 🔑 [获取API Key](https://dashscope.console.aliyun.com/apiKey)

    ### Jina AI
    - 📖 [官方文档](https://jina.ai/embeddings/)
    - 💻 [GitHub](https://github.com/jina-ai/jina)
    - 🔑 [获取API Key](https://jina.ai/embeddings/)

    ### Cohere
    - 📖 [官方文档](https://docs.cohere.com/docs/embeddings)
    - 💰 [定价](https://cohere.com/pricing)
    - 🔑 [获取API Key](https://dashboard.cohere.com/api-keys)

    ### Sentence Transformers
    - 📖 [官方文档](https://www.sbert.net/)
    - 💻 [GitHub](https://github.com/UKPLab/sentence-transformers)
    - 🤗 [模型库](https://huggingface.co/sentence-transformers)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📝 总结

    ### 🎯 核心要点

    1. **API服务优势**
       - ✅ 快速上线，无需部署
       - ✅ 自动更新，无需维护
       - ✅ 高可用性，SLA保障

    2. **本地部署优势**
       - ✅ 数据隐私，完全可控
       - ✅ 成本低廉，大规模优势
       - ✅ 低延迟，离线可用

    3. **选择建议**
       - 🚀 **原型阶段** → API服务
       - 💼 **小规模生产** → API服务
       - 🏭 **大规模生产** → 本地部署
       - 🔒 **数据敏感** → 本地部署

    4. **最佳实践**
       - 批量处理提高效率
       - 缓存结果避免重复计算
       - 错误处理和重试机制
       - 混合策略兼顾优势

    ### 🚀 快速开始

    **使用OpenAI**:
    ```python
    from openai import OpenAI
    client = OpenAI(api_key="sk-...")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=["你的文本"]
    )
    ```

    **使用本地模型**:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(["你的文本"])
    ```

    ### 💡 下一步

    1. 根据需求选择合适的方案
    2. 申请API密钥或下载本地模型
    3. 在小规模数据上测试
    4. 评估性能和成本
    5. 部署到生产环境
    """
    )
    return


if __name__ == "__main__":
    app.run()
