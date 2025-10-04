"""
Sentence Transformers 完全指南

Sentence Transformers是一个Python库，用于生成句子、段落和图像的语义向量表示。
它基于Transformer模型（如BERT、RoBERTa等），可以轻松地将文本转换为固定长度的向量。

特点：
1. 🚀 简单易用 - 几行代码即可生成高质量向量
2. 🎯 预训练模型 - 提供多种预训练模型
3. 🌐 多语言支持 - 支持100+种语言
4. 📊 高性能 - 优化的推理速度
5. 🔧 可微调 - 支持自定义数据集微调

作者: Marimo Notebook
日期: 2025-01-XX
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="Sentence Transformers 完全指南")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🤖 Sentence Transformers 完全指南

    ## 什么是Sentence Transformers？

    **Sentence Transformers** 是一个用于生成句子和段落嵌入（embeddings）的Python框架。它可以将文本转换为密集向量表示，这些向量可以用于：

    - 🔍 **语义搜索** - 找到语义相似的文本
    - 📊 **文本聚类** - 将相似文本分组
    - 🎯 **文本分类** - 基于向量的分类任务
    - 🔗 **问答系统** - 匹配问题和答案
    - 🌐 **跨语言检索** - 多语言文本匹配

    ### 核心优势

    1. **简单易用** - 3行代码即可生成向量
    2. **高质量** - 基于SOTA的Transformer模型
    3. **快速** - 优化的推理性能
    4. **灵活** - 支持自定义模型和微调
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📦 安装

    ```bash
    # 基础安装
    pip install sentence-transformers

    # 或使用uv
    uv pip install sentence-transformers
    ```

    **依赖项**:
    - PyTorch >= 1.11.0
    - transformers >= 4.34.0
    - tqdm
    - numpy
    - scikit-learn
    """
    )
    return


@app.cell
def _():
    # 📦 导入必要的库
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    
    print("=" * 60)
    print("🤖 Sentence Transformers 导入成功")
    print("=" * 60)
    
    return SentenceTransformer, np, util


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎯 快速开始：加载模型

    Sentence Transformers提供了多种预训练模型，适用于不同的场景。
    """
    )
    return


@app.cell
def _(SentenceTransformer):
    # 🎯 加载预训练模型
    # 使用多语言模型（支持中文）
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("✅ 模型加载成功")
    print(f"📊 模型名称: {model._model_card_vars.get('model_name', 'N/A')}")
    print(f"📏 向量维度: {model.get_sentence_embedding_dimension()}")
    print(f"🔢 最大序列长度: {model.max_seq_length}")
    
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📝 基础操作：文本编码

    将文本转换为向量是最基础的操作。
    """
    )
    return


@app.cell
def _(model, np):
    # 📝 示例1: 单个句子编码
    sentence = "人工智能正在改变世界"
    embedding = model.encode(sentence)
    
    print("=" * 60)
    print("📝 单句编码")
    print("=" * 60)
    print(f"原文: {sentence}")
    print(f"向量维度: {embedding.shape}")
    print(f"向量前5个值: {embedding[:5]}")
    print()
    
    # 📝 示例2: 批量编码
    sentences = [
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "今天天气真不错",
        "我喜欢吃披萨"
    ]
    
    embeddings = model.encode(sentences)
    
    print("=" * 60)
    print("📝 批量编码")
    print("=" * 60)
    print(f"句子数量: {len(sentences)}")
    print(f"向量矩阵形状: {embeddings.shape}")
    print()

    # 显示每个句子
    for num, text in enumerate(sentences):
        print(f"{num + 1}. {text}")

    return embedding, embeddings, sentence, sentences


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔍 语义相似度计算

    计算文本之间的语义相似度是最常见的应用。
    """
    )
    return


@app.cell
def _(embeddings, model, sentences, util):
    # 🔍 计算余弦相似度
    print("=" * 60)
    print("🔍 语义相似度矩阵")
    print("=" * 60)
    
    # 计算所有句子对之间的相似度
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    print("\n相似度矩阵 (0-1之间，越接近1越相似):\n")

    # 打印表头
    print("     ", end="")
    for col_idx in range(len(sentences)):
        print(f"  句{col_idx+1}  ", end="")
    print()

    # 打印相似度矩阵
    for row_idx, sent in enumerate(sentences):
        print(f"句{row_idx+1} ", end="")
        for col_idx in range(len(sentences)):
            score = cosine_scores[row_idx][col_idx].item()
            print(f" {score:.3f} ", end="")
        print(f" | {sent[:15]}...")

    print()

    # 找出最相似的句子对
    print("🎯 最相似的句子对:")
    pairs_list = []
    for pair_i in range(len(sentences)):
        for pair_j in range(pair_i + 1, len(sentences)):
            score = cosine_scores[pair_i][pair_j].item()
            pairs_list.append((pair_i, pair_j, score))

    # 排序并显示前3对
    pairs_list.sort(key=lambda x: x[2], reverse=True)
    for rank_num, (i, j, score) in enumerate(pairs_list[:3], 1):
        print(f"{rank_num}. 相似度: {score:.4f}")
        print(f"   句子A: {sentences[i]}")
        print(f"   句子B: {sentences[j]}")
        print()

    return cosine_scores, pairs_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔎 实战案例1：语义搜索

    给定一个查询，从文档库中找到最相关的文档。
    """
    )
    return


@app.cell
def _(model, util):
    # 🔎 语义搜索示例
    print("=" * 60)
    print("🔎 语义搜索示例")
    print("=" * 60)
    
    # 文档库
    documents = [
        "Python是一种高级编程语言",
        "机器学习需要大量的数据",
        "深度学习是机器学习的子集",
        "神经网络模拟人脑的工作方式",
        "自然语言处理处理人类语言",
        "计算机视觉让机器理解图像",
        "强化学习通过奖励来学习",
        "数据科学结合统计学和编程"
    ]
    
    # 编码文档
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    
    # 查询
    query = "什么是深度学习？"
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # 计算相似度
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=3)[0]
    
    print(f"\n查询: {query}\n")
    print("🎯 最相关的文档:\n")
    
    for rank, hit in enumerate(hits, 1):
        doc_id = hit['corpus_id']
        score_val = hit['score']
        print(f"{rank}. 相似度: {score_val:.4f}")
        print(f"   文档: {documents[doc_id]}")
        print()
    
    return doc_embeddings, doc_id, documents, hits, query, query_embedding, rank, score_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📊 实战案例2：文本聚类

    将相似的文本自动分组。
    """
    )
    return


@app.cell
def _(model, np):
    # 📊 文本聚类示例
    from sklearn.cluster import KMeans
    
    print("=" * 60)
    print("📊 文本聚类示例")
    print("=" * 60)
    
    # 准备文本数据
    texts = [
        # 科技类
        "人工智能的发展",
        "机器学习算法",
        "深度神经网络",
        # 体育类
        "足球比赛结果",
        "篮球运动员",
        "奥运会金牌",
        # 美食类
        "中国传统美食",
        "意大利披萨",
        "日本寿司"
    ]
    
    # 编码
    text_embeddings = model.encode(texts)
    
    # K-means聚类
    num_clusters = 3
    clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
    clustering_model.fit(text_embeddings)
    cluster_assignment = clustering_model.labels_
    
    # 按类别组织结果
    clustered_texts = {}
    for text_idx, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_texts:
            clustered_texts[cluster_id] = []
        clustered_texts[cluster_id].append(texts[text_idx])
    
    # 显示结果
    print(f"\n将 {len(texts)} 个文本分为 {num_clusters} 个类别:\n")
    
    for cluster_num, cluster_texts in sorted(clustered_texts.items()):
        print(f"📁 类别 {cluster_num + 1}:")
        for text_item in cluster_texts:
            print(f"   - {text_item}")
        print()
    
    return (
        KMeans,
        cluster_assignment,
        cluster_id,
        cluster_num,
        cluster_texts,
        clustered_texts,
        clustering_model,
        num_clusters,
        text_embeddings,
        text_idx,
        text_item,
        texts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎨 实战案例3：问答匹配

    找到与问题最匹配的答案。
    """
    )
    return


@app.cell
def _(model, util):
    # 🎨 问答匹配示例
    print("=" * 60)
    print("🎨 问答匹配系统")
    print("=" * 60)
    
    # 问答对
    qa_pairs = {
        "Python是什么？": "Python是一种高级、解释型、通用的编程语言。",
        "如何学习机器学习？": "学习机器学习需要掌握数学基础、编程技能和实践经验。",
        "什么是深度学习？": "深度学习是机器学习的一个分支，使用多层神经网络。",
        "如何提高代码质量？": "通过代码审查、单元测试和遵循最佳实践来提高代码质量。",
        "什么是API？": "API是应用程序编程接口，允许不同软件之间通信。"
    }
    
    questions_list = list(qa_pairs.keys())
    answers_list = list(qa_pairs.values())
    
    # 编码问题
    question_embeddings = model.encode(questions_list, convert_to_tensor=True)
    
    # 用户问题
    user_questions = [
        "Python编程语言是什么",
        "深度学习的定义",
        "怎样写出好的代码"
    ]
    
    print("\n🔍 用户问题匹配:\n")
    
    for user_q in user_questions:
        user_q_embedding = model.encode(user_q, convert_to_tensor=True)
        search_hits = util.semantic_search(user_q_embedding, question_embeddings, top_k=1)[0]
        
        best_match = search_hits[0]
        matched_q_idx = best_match['corpus_id']
        similarity = best_match['score']
        
        print(f"❓ 用户问题: {user_q}")
        print(f"✅ 匹配问题: {questions_list[matched_q_idx]}")
        print(f"💡 答案: {answers_list[matched_q_idx]}")
        print(f"📊 相似度: {similarity:.4f}")
        print()
    
    return (
        answers_list,
        best_match,
        matched_q_idx,
        qa_pairs,
        question_embeddings,
        questions_list,
        search_hits,
        similarity,
        user_q,
        user_q_embedding,
        user_questions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🌐 实战案例4：跨语言语义搜索

    使用多语言模型进行跨语言检索。
    """
    )
    return


@app.cell
def _(model, util):
    # 🌐 跨语言搜索示例
    print("=" * 60)
    print("🌐 跨语言语义搜索")
    print("=" * 60)

    # 多语言文档库
    multilingual_docs = [
        "Machine learning is a subset of artificial intelligence",  # 英文
        "人工智能正在改变世界",  # 中文
        "El aprendizaje profundo utiliza redes neuronales",  # 西班牙语
        "深度学习需要大量的训练数据",  # 中文
        "Natural language processing helps computers understand human language",  # 英文
    ]

    # 编码文档
    multi_doc_embeddings = model.encode(multilingual_docs, convert_to_tensor=True)

    # 中文查询
    chinese_query = "什么是机器学习"
    query_emb = model.encode(chinese_query, convert_to_tensor=True)

    # 搜索
    results = util.semantic_search(query_emb, multi_doc_embeddings, top_k=3)[0]

    print(f"\n🔍 查询 (中文): {chinese_query}\n")
    print("🎯 跨语言搜索结果:\n")

    for position, result in enumerate(results, 1):
        doc_idx = result['corpus_id']
        sim_score = result['score']
        print(f"{position}. 相似度: {sim_score:.4f}")
        print(f"   文档: {multilingual_docs[doc_idx]}")
        print()

    return (
        chinese_query,
        doc_idx,
        multi_doc_embeddings,
        multilingual_docs,
        position,
        query_emb,
        result,
        results,
        sim_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## ⚙️ 高级功能：编码参数

    `encode()` 方法支持多种参数来优化性能和结果。
    """
    )
    return


@app.cell
def _(model):
    # ⚙️ 编码参数示例
    print("=" * 60)
    print("⚙️ 编码参数演示")
    print("=" * 60)

    sample_texts = [
        "这是第一个句子",
        "这是第二个句子"
    ]

    # 1. 基础编码（返回numpy数组）
    basic_emb = model.encode(sample_texts)
    print(f"\n1️⃣ 基础编码:")
    print(f"   类型: {type(basic_emb)}")
    print(f"   形状: {basic_emb.shape}")

    # 2. 转换为Tensor（用于PyTorch）
    tensor_emb = model.encode(sample_texts, convert_to_tensor=True)
    print(f"\n2️⃣ Tensor编码:")
    print(f"   类型: {type(tensor_emb)}")
    print(f"   形状: {tensor_emb.shape}")

    # 3. 归一化向量
    normalized_emb = model.encode(sample_texts, normalize_embeddings=True)
    print(f"\n3️⃣ 归一化编码:")
    print(f"   向量长度: {np.linalg.norm(normalized_emb[0]):.4f} (应该接近1.0)")

    # 4. 批处理大小
    large_batch = ["句子" + str(num) for num in range(100)]
    batch_emb = model.encode(large_batch, batch_size=32, show_progress_bar=True)
    print(f"\n4️⃣ 批处理编码:")
    print(f"   处理了 {len(large_batch)} 个句子")
    print(f"   批大小: 32")

    return (
        basic_emb,
        batch_emb,
        large_batch,
        normalized_emb,
        num,
        sample_texts,
        tensor_emb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎯 常用预训练模型

    不同的模型适用于不同的场景。
    """
    )
    return


@app.cell
def _():
    # 🎯 常用模型列表
    print("=" * 60)
    print("🎯 推荐的预训练模型")
    print("=" * 60)

    models_info = [
        {
            "name": "all-MiniLM-L6-v2",
            "lang": "英文",
            "dim": 384,
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐⭐",
            "use_case": "通用英文任务，速度优先"
        },
        {
            "name": "all-mpnet-base-v2",
            "lang": "英文",
            "dim": 768,
            "speed": "⚡⚡",
            "quality": "⭐⭐⭐⭐⭐",
            "use_case": "高质量英文任务"
        },
        {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "lang": "多语言",
            "dim": 384,
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐⭐⭐",
            "use_case": "多语言任务，包括中文"
        },
        {
            "name": "paraphrase-multilingual-mpnet-base-v2",
            "lang": "多语言",
            "dim": 768,
            "speed": "⚡⚡",
            "quality": "⭐⭐⭐⭐⭐",
            "use_case": "高质量多语言任务"
        },
        {
            "name": "distiluse-base-multilingual-cased-v2",
            "lang": "多语言",
            "dim": 512,
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐⭐⭐",
            "use_case": "平衡速度和质量"
        }
    ]

    print("\n模型对比:\n")
    for model_info in models_info:
        print(f"📦 {model_info['name']}")
        print(f"   语言: {model_info['lang']}")
        print(f"   维度: {model_info['dim']}")
        print(f"   速度: {model_info['speed']}")
        print(f"   质量: {model_info['quality']}")
        print(f"   用途: {model_info['use_case']}")
        print()

    return model_info, models_info


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🚀 性能优化技巧

    提高编码速度和效率的方法。
    """
    )
    return


@app.cell
def _(model):
    import time

    # 🚀 性能优化示例
    print("=" * 60)
    print("🚀 性能优化技巧")
    print("=" * 60)

    test_sentences = ["测试句子 " + str(n) for n in range(1000)]

    # 1. 小批量 vs 大批量
    print("\n1️⃣ 批量大小对比:\n")

    start = time.time()
    _ = model.encode(test_sentences, batch_size=8, show_progress_bar=False)
    time_small = time.time() - start
    print(f"   批大小=8:  {time_small:.2f}秒")

    start = time.time()
    _ = model.encode(test_sentences, batch_size=64, show_progress_bar=False)
    time_large = time.time() - start
    print(f"   批大小=64: {time_large:.2f}秒")
    print(f"   提速: {time_small/time_large:.2f}x")

    # 2. 归一化的影响
    print("\n2️⃣ 归一化对比:\n")

    start = time.time()
    _ = model.encode(test_sentences[:100], normalize_embeddings=False, show_progress_bar=False)
    time_no_norm = time.time() - start
    print(f"   不归一化: {time_no_norm:.3f}秒")

    start = time.time()
    _ = model.encode(test_sentences[:100], normalize_embeddings=True, show_progress_bar=False)
    time_norm = time.time() - start
    print(f"   归一化:   {time_norm:.3f}秒")

    # 3. 最佳实践建议
    print("\n3️⃣ 最佳实践:\n")
    print("   ✅ 使用较大的batch_size（如32-64）")
    print("   ✅ 对于相似度计算，使用normalize_embeddings=True")
    print("   ✅ 使用convert_to_tensor=True避免类型转换")
    print("   ✅ 预先编码静态文档库，避免重复计算")
    print("   ✅ 使用GPU加速（如果可用）")

    return n, start, test_sentences, time, time_large, time_no_norm, time_norm, time_small


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 主要API参考

    以下是Sentence Transformers的核心API总结。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 🔧 核心类和方法

    | API | 说明 | 参数 | 返回值 |
    |-----|------|------|--------|
    | `SentenceTransformer(model_name)` | 加载预训练模型 | `model_name`: 模型名称或路径<br>`device`: 设备('cuda'/'cpu')<br>`cache_folder`: 缓存目录 | SentenceTransformer对象 |
    | `model.encode(sentences)` | 将文本编码为向量 | `sentences`: 字符串或列表<br>`batch_size`: 批大小(默认32)<br>`show_progress_bar`: 显示进度<br>`convert_to_tensor`: 返回Tensor<br>`normalize_embeddings`: 归一化 | numpy数组或Tensor |
    | `model.get_sentence_embedding_dimension()` | 获取向量维度 | 无 | int |
    | `model.max_seq_length` | 最大序列长度 | 无 | int |
    | `model.tokenize(texts)` | 文本分词 | `texts`: 字符串或列表 | 字典(input_ids, attention_mask等) |

    ### 🔍 相似度计算工具

    | API | 说明 | 参数 | 返回值 |
    |-----|------|------|--------|
    | `util.cos_sim(a, b)` | 计算余弦相似度 | `a`: 向量或矩阵<br>`b`: 向量或矩阵 | 相似度矩阵(Tensor) |
    | `util.dot_score(a, b)` | 计算点积分数 | `a`: 向量或矩阵<br>`b`: 向量或矩阵 | 分数矩阵(Tensor) |
    | `util.semantic_search(query, corpus)` | 语义搜索 | `query`: 查询向量<br>`corpus`: 文档向量<br>`top_k`: 返回前k个结果<br>`score_function`: 评分函数 | 列表[{'corpus_id': int, 'score': float}] |
    | `util.paraphrase_mining(model, sentences)` | 挖掘相似句对 | `model`: 模型对象<br>`sentences`: 句子列表<br>`top_k`: 返回前k对 | 列表[(score, idx1, idx2)] |

    ### 📊 评估和训练

    | API | 说明 | 参数 | 返回值 |
    |-----|------|------|--------|
    | `model.similarity(sentences1, sentences2)` | 计算两组句子的相似度 | `sentences1`: 句子列表<br>`sentences2`: 句子列表 | 相似度矩阵 |
    | `model.save(path)` | 保存模型 | `path`: 保存路径 | 无 |
    | `SentenceTransformer.load(path)` | 加载模型 | `path`: 模型路径 | SentenceTransformer对象 |

    ### 🎯 常用参数说明

    | 参数 | 类型 | 默认值 | 说明 |
    |------|------|--------|------|
    | `batch_size` | int | 32 | 批处理大小，越大速度越快但内存占用越高 |
    | `show_progress_bar` | bool | True | 是否显示进度条 |
    | `convert_to_tensor` | bool | False | 是否返回PyTorch Tensor |
    | `normalize_embeddings` | bool | False | 是否归一化向量（推荐用于相似度计算） |
    | `device` | str | None | 设备选择：'cuda'、'cpu'或None(自动) |
    | `num_workers` | int | 0 | 数据加载的工作进程数 |

    ### 🌟 推荐模型列表

    | 模型名称 | 语言 | 维度 | 速度 | 质量 | 适用场景 |
    |---------|------|------|------|------|---------|
    | `all-MiniLM-L6-v2` | 英文 | 384 | ⚡⚡⚡ | ⭐⭐⭐ | 通用英文，速度优先 |
    | `all-mpnet-base-v2` | 英文 | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ | 高质量英文任务 |
    | `paraphrase-multilingual-MiniLM-L12-v2` | 多语言 | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ | 多语言（含中文） |
    | `paraphrase-multilingual-mpnet-base-v2` | 多语言 | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ | 高质量多语言 |
    | `distiluse-base-multilingual-cased-v2` | 多语言 | 512 | ⚡⚡⚡ | ⭐⭐⭐⭐ | 平衡速度和质量 |
    | `msmarco-distilbert-base-v4` | 英文 | 768 | ⚡⚡ | ⭐⭐⭐⭐ | 信息检索 |
    | `multi-qa-MiniLM-L6-cos-v1` | 英文 | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ | 问答系统 |

    ### 💡 使用技巧

    1. **选择合适的模型**
       - 英文任务：`all-mpnet-base-v2`（质量）或`all-MiniLM-L6-v2`（速度）
       - 中文/多语言：`paraphrase-multilingual-mpnet-base-v2`
       - 问答系统：`multi-qa-*`系列

    2. **性能优化**
       - 增大`batch_size`（32-128）
       - 使用GPU：`device='cuda'`
       - 预先编码静态文档
       - 使用`normalize_embeddings=True`简化相似度计算

    3. **相似度计算**
       - 归一化向量后，余弦相似度 = 点积
       - 使用`util.semantic_search()`进行高效搜索
       - 阈值建议：>0.7为高相似，0.5-0.7为中等，<0.5为低相似

    4. **内存管理**
       - 大规模数据分批处理
       - 使用`convert_to_tensor=False`节省内存
       - 及时释放不需要的向量
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔗 资源链接

    ### 官方资源
    - 📖 [官方文档](https://www.sbert.net/)
    - 💻 [GitHub仓库](https://github.com/UKPLab/sentence-transformers)
    - 🤗 [HuggingFace模型库](https://huggingface.co/sentence-transformers)
    - 📊 [预训练模型列表](https://www.sbert.net/docs/pretrained_models.html)

    ### 学习资源
    - 📚 [入门教程](https://www.sbert.net/docs/quickstart.html)
    - 🎓 [示例代码](https://github.com/UKPLab/sentence-transformers/tree/master/examples)
    - 📝 [论文](https://arxiv.org/abs/1908.10084)

    ### 社区
    - 💬 [GitHub Discussions](https://github.com/UKPLab/sentence-transformers/discussions)
    - 🐛 [Issue Tracker](https://github.com/UKPLab/sentence-transformers/issues)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📝 总结

    **Sentence Transformers** 是一个强大而易用的文本向量化库：

    ✅ **优点**:
    - 简单易用，3行代码即可开始
    - 丰富的预训练模型
    - 优秀的多语言支持
    - 高性能推理
    - 活跃的社区支持

    🎯 **适用场景**:
    - 语义搜索和信息检索
    - 文本聚类和分类
    - 问答系统
    - 推荐系统
    - 重复检测

    💡 **快速开始**:
    ```python
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(['你好世界', 'Hello World'])
    ```

    🚀 **下一步**:
    - 尝试不同的预训练模型
    - 在自己的数据上微调模型
    - 集成到实际应用中
    """
    )
    return


if __name__ == "__main__":
    app.run()


