import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        r"""
        # 向量嵌入：从零到英雄（使用Python和LangChain）

        ![预览图片](https://miro.medium.com/v2/resize:fit:700/0*gKq85CDJr0zJ50oN)

        **作者：** [Vamshi Krishna Ginna](https://medium.com/@vamshiginna1606)  
        **发布时间：** 2025年5月31日（更新：2025年5月31日）

        ---

        ## 1. 引言

        在当今AI驱动的世界中，机器需要的不仅仅是原始文本——它们需要理解。这就是**向量嵌入**发挥作用的地方。这些强大的数值表示将单词、句子，甚至完整的文档转换为捕获含义、上下文和关系的高维向量。

        无论您是在构建**语义搜索引擎**、**推荐系统**，还是**真正理解您的聊天机器人**，嵌入都是基础。它们允许AI模型测量相似性、检测细微差别，并以传统数据库根本无法做到的方式连接想法。

        > **将嵌入视为人类语言和机器推理之间的桥梁。**

        这篇博客将带您从嵌入的**绝对基础**到使用Python构建**基于LangChain的实践助手**。我们将探索：

        - 嵌入如何工作
        - 它们在哪里使用
        - 传统数据库和向量数据库之间的区别
        - 以及如何使用**OpenAI**、**HuggingFace**、**FAISS**和**LangChain**等工具将所有内容整合在一起

        📌 **奖励：** 如果您是LangChain的新手或对构建完整的检索增强生成（RAG）应用程序感兴趣，请查看我之前的文章：[LangChain速成课程——第2部分：构建您的第一个RAG应用](https://medium.com/@vamshiginna1606/langchain-crash-course-part-2-build-your-first-rag-app-24908b14d337)

        让我们深入了解并揭开嵌入的神秘面纱——从零到英雄。
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. NLP先决条件

        在深入研究向量嵌入之前，了解一些关键的NLP概念是必要的，这些概念构成了机器处理和理解文本的基础。

        ![NLP基础](https://miro.medium.com/v2/resize:fit:700/0*nvFxQ3-lGWh-pKs_)

        ### 2.1. 分词（Tokenization）

        分词将句子或文档分解为称为标记的较小单位——通常是单词或子词。

        ```python
        from nltk.tokenize import word_tokenize
        text = "Vector embeddings are powerful!"
        tokens = word_tokenize(text)
        print(tokens)
        ```

        输出：
        ```
        ['Vector', 'embeddings', 'are', 'powerful', '!']
        ```

        ### 2.2. 停用词（Stopwords）

        停用词是像"and"、"the"、"is"这样的常见词，它们携带的含义很少，在预处理中经常被删除。

        ```python
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
        print(filtered_tokens)
        ```

        输出：
        ```
        ['Vector', 'embeddings', 'powerful', '!']
        ```

        ### 2.3. 词形还原（Lemmatization）

        词形还原使用上下文和词汇将单词还原为其基本形式。

        ```python
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        print(lemmatizer.lemmatize("running"))
        ```

        ### 2.4. 词袋模型（Bag of Words - BoW）

        一个简单的模型，将文本表示为词计数的向量。

        ```python
        from sklearn.feature_extraction.text import CountVectorizer
        corpus = ["Embeddings convert text into vectors.", 
                  "Vectors can be compared mathematically."]
        vectorizer = CountVectorizer()
        print(vectorizer.fit_transform(corpus).toarray())
        ```

        ### 2.5. TF-IDF（词频-逆文档频率）

        根据单词对语料库中文档的重要性对单词进行加权。

        ```python
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        print(vectorizer.fit_transform(corpus).toarray())
        ```

        这些基础工具将帮助您理解原始文本如何演变为结构化的数值表示——为更深层的嵌入概念奠定基础。
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. 从文本到向量

        现代NLP的核心是一个简单的想法：**将文本转换为数字**。为什么？因为机器不理解单词——它们理解**向量**（即数字数组）。这就是嵌入发挥作用的地方。

        ### 3.1. 嵌入的主要目标

        将自然语言转换为捕获输入含义和上下文的**密集向量表示**。

        ### 3.2. 独热编码：老式方法

        在嵌入之前，我们使用**独热编码**。它将每个单词表示为只有一个'1'和其余'0'的二进制向量。

        示例：
        ```python
        # 词汇表：["king", "queen", "man", "woman"]
        # "king" → [1, 0, 0, 0]
        ```

        **局限性：**
        - 不能捕获单词之间的任何关系
        - 导致高维稀疏向量

        ### 3.3. 嵌入：现代方法

        嵌入不使用二进制向量，而是**为每个单词分配一个密集的固定长度向量**。这些向量从数据中学习并捕获语义关系。

        示例：
        ```python
        # "king" → [0.25, 0.78, -0.39, ...]  (通常300–3072维)
        ```

        这些嵌入实现了**语义算术**：
        - `king - man + woman ≈ queen`

        ### 3.4. 相同长度，不同文本

        嵌入最强大的特性之一是**任何句子，无论多长，都会转换为相同长度的向量**。长度由模型决定（例如，MiniLM为384，OpenAI的`text-embedding-3-large`为3072）。

        ### 3.5. 为什么重要

        这种转换让我们能够：
        - 使用余弦距离测量文本之间的**相似性**
        - 构建**搜索引擎**和**推荐系统**
        - 为深度学习模型提供有意义的输入
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 4. 嵌入的类型

        并非所有嵌入都是相同的。根据任务的不同，不同类型的嵌入有助于在不同的粒度级别上表示语言。

        ### 4.1. 词嵌入

        每个单词都映射到一个唯一的向量。基于在大型语料库中的使用捕获含义。

        - **常见模型：** **Word2Vec**、**GloVe**、**FastText**
        - **适用于：** 类比任务、情感分析
        - **局限性：** 单词"bank"（河岸vs银行）使用相同向量

        ```python
        from gensim.models import Word2Vec
        sentences = [["king", "queen", "man", "woman"]]
        model = Word2Vec(sentences, min_count=1)
        print(model.wv["king"])
        ```

        ### 4.2. 句子嵌入

        将整个句子编码为捕获整体含义的单个向量。

        - **常见模型：** **SBERT**、**Universal Sentence Encoder**
        - **适用于：** 语义搜索、重复检测
        - **优势：** 处理词序、标点符号和含义

        ```python
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(model.encode("Vector embeddings are powerful."))
        ```

        ### 4.3. 文档嵌入

        超越句子，捕获完整文档的上下文。

        - **常见模型：** **Doc2Vec**、**DPR（Dense Passage Retriever）**
        - **适用于：** 大规模文档相似性、分类

        ### 4.4. 多模态嵌入

        将图像、音频和文本映射到同一向量空间。

        - **用于：** **CLIP**、**DALL·E**、**Flamingo**
        - **适用于：** 跨模态搜索（例如，图像字幕、视觉问答）

        每种类型的嵌入在解决不同的现实世界问题中都发挥作用。接下来，让我们使用**LangChain和OpenAI嵌入进行实践演示**。
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 5. LangChain嵌入（使用OpenAI和Hugging Face实践）

        让我们实践一下！我们将使用**LangChain**，这是一个使处理LLM和向量存储变得非常简单的框架。首先，我们将使用**OpenAI**和**Hugging Face**生成嵌入，然后比较它们。

        ### 5.1. 设置环境

        安装依赖项并加载环境变量。

        ```bash
        pip install langchain-openai langchain-huggingface python-dotenv
        ```

        ```python
        import os
        from dotenv import load_dotenv
        load_dotenv()  # 从.env加载API密钥
        ```

        > 按照此指南生成API密钥：[如何设置OpenAI、Groq和LangSmith API密钥](https://medium.com/@vamshiginna1606/api-key-setup-for-openai-groq-and-langsmith-in-your-projects-edf745e9507c)

        ### 5.2. 使用LangChain的OpenAI嵌入

        ```python
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        text = "This is a blog post on vector embeddings."
        embeddings_result = embeddings.embed_query(text)

        print(f"Length: {len(embeddings_result)}")
        print(f"Type: {type(embeddings_result)}")
        ```

        输出：
        ```
        Length: 3072
        Type: <class 'list'>
        ```

        - 任何输入文本的向量长度相同
        - 非常适合语义搜索或向量相似性

        ### 5.3. 维度缩减（可选）

        您可以使用LangChain的内置配置减少维度：

        ```python
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
        ```

        您会失去一些粒度，但这有助于存储和速度。

        ### 5.4. Hugging Face嵌入（开源）

        您可以在Hugging Face上创建免费账户，并在设置下生成新的访问密钥，并将该密钥包含在您的.env文件中作为`HF_TOKEN=hf_vCixxxxxxxxxxxxxxxxxxxxx`

        ```python
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text = "This is a blog post on vector embeddings."
        embeddings_result = embeddings.embed_query(text)

        print(f"Length: {len(embeddings_result)}")
        ```

        - 输出：384维向量
        - 免费使用，非常适合本地测试
        - 速度和质量的良好平衡
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 6. 理解嵌入向量属性

        现在我们已经使用LangChain生成了嵌入，让我们解析这些向量实际意味着什么以及它们如何在现实世界应用中使用。

        ### 6.1. 固定长度，高维度

        每个模型生成**固定长度**的向量，无论输入长度如何：

        - OpenAI（`text-embedding-3-large`）→ 3072维
        - Hugging Face（`MiniLM-L6-v2`）→ 384维

        这种一致性允许轻松比较和存储不同的文本输入。

        ### 6.2. 嵌入作为特征表示

        向量中的每个元素可以被认为代表文本的**潜在特征**——如语调、主题、语法或情感。

        ![特征表示](https://miro.medium.com/v2/resize:fit:700/0*Jw4-wiq53fZsdZQ9)

        ```python
        print(type(embeddings_result))  # <class 'list'>
        print(len(embeddings_result))   # 3072 or 384
        ```

        这些特征从大量数据集中学习，并压缩到一个空间中，其中相似的含义**更接近**。

        ### 6.3. 维度缩减（可选但有用）

        为了提高速度和效率（例如，在移动设备或内存应用中），您可以减少向量维度而不会在准确性上损失太多。

        ```python
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
        ```

        > 提示：这在比较来自不同模型的向量或在轻量级应用中嵌入多个句子时很有用。

        ### 6.4. 用例：语义比较

        您现在可以应用相似性度量来比较两个向量的接近程度：

        ```python
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import euclidean_distances

        documents = [
            "What is the capital of USA?",
            "Who is the president of USA?",
            "Who is the Prime Minister of India?",
        ]

        my_query = "Narendra Modi is the Prime Minister of India."
        query_embedding = embeddings.embed_query(my_query)
        documents_embeddings = embeddings.embed_documents(documents)

        cosine_similarity([query_embedding], documents_embeddings)
        # 输出: array([[0.13519943, 0.28759853, 0.729995  ]])

        euclidean_distances([query_embedding], documents_embeddings)
        # 输出: array([[1.31514299, 1.19365109, 0.73485375]])
        ```

        - **余弦相似性** → 测量角度（适用于文本）
        - **欧几里得距离** → 测量绝对距离（在NLP中不太常见）

        ![相似性度量](https://miro.medium.com/v2/resize:fit:700/1*y7dlnXhXjhT3WO2t9ZyRFg.png)

        接下来：我们将探索**向量数据库与传统数据库的区别**，以及为什么您需要专门的系统来高效处理嵌入。
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 7. 向量数据库 vs SQL/NoSQL数据库

        传统数据库（SQL或NoSQL）非常适合存储和检索结构化数据，如数字、字符串和关系。但是当涉及到**语义搜索或相似性搜索**时，它们就不够用了。

        这就是**向量数据库**发挥作用的地方。

        ### 7.1. 传统数据库

        ![传统数据库](https://miro.medium.com/v2/resize:fit:700/1*9uHb_WLiciKZSVrITLAmfA.png)

        这些系统使用表或类似JSON的文档存储**结构化**或**半结构化**数据。但它们无法原生执行向量相似性操作，如余弦相似性。

        ### 7.2. 向量数据库

        向量数据库存储**嵌入**并允许高效的**最近邻搜索**。它们针对以下方面进行了优化：

        - **语义搜索**
        - **推荐引擎**
        - **聚类和异常检测**

        ![向量数据库](https://miro.medium.com/v2/resize:fit:700/1*YIiAq2A7fOtZEjsKel5Z1Q.png)

        ### 7.3. 为什么使用向量数据库？

        1. **快速相似性搜索** 使用近似最近邻（ANN）算法，如HNSW或IVF，实现毫秒级检索。
        2. **元数据过滤** 将相似性与价格范围、类别、标签等过滤器结合。
        3. **可扩展和分布式** 云向量数据库可扩展到数百万条记录。
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 8. 向量数据库中的索引和相似性搜索

        向量搜索不仅仅是将每个向量与其他每个向量进行比较——那样会太慢。相反，向量数据库使用**索引**技术使相似性搜索变得快如闪电且可扩展。

        ### 8.1. 什么是向量索引？

        **索引**是一种数据结构，可以在向量集合上实现快速搜索。没有它，每次搜索都需要将查询与数据库中的每个项目进行比较（O(n)时间）。

        ### 8.2. 向量索引的类型

        #### 8.2.1 平面索引（暴力搜索）

        - 检查与每个存储向量的相似性
        - 准确，但对大数据集来说很慢
        - 最适合小数据集或测试

        ```python
        # FAISS平面索引
        import faiss
        index = faiss.IndexFlatL2(384)  # 对于384维向量
        index.add(embedding_matrix)
        ```

        #### 8.2.2. HNSW（分层可导航小世界）

        - 基于图的结构
        - 非常快速和可扩展
        - 略微近似，但准确性高

        > 在Weaviate、Qdrant和FAISS的`IndexHNSWFlat`中使用

        #### 8.2.3. IVF（倒排文件索引）

        - 将向量聚类成组（如k-means）
        - 首先在几个聚类中搜索
        - 更快，但需要训练和调优

        > 在FAISS的`IndexIVFFlat`中使用

        ### 8.4. 相似性度量

        - **余弦相似性** → 最适合文本
        - **欧几里得距离** → 适合空间数据
        - **点积** → 在一些深度学习模型中使用

        ```python
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_similarity([query_vector], [doc_vector])
        ```

        ![相似性度量](https://miro.medium.com/v2/resize:fit:700/1*fTP9DL_OKynWzteATPiBVw.png)

        接下来，让我们实现一个**简单的FAISS驱动的搜索系统**
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 9. 迷你项目：使用LangChain + OpenAI + FAISS的语义搜索

        让我们构建一个**搜索界面**，用户输入查询，应用程序使用嵌入和FAISS从小数据集中返回语义上最相似的句子。

        这个迷你项目展示了如何：

        1. 通过LangChain使用OpenAI将文本转换为嵌入
        2. 将它们存储在FAISS向量索引中
        3. 执行相似性搜索

        ### 步骤1：设置

        安装所需的库：

        ```bash
        pip install langchain faiss-cpu openai python-dotenv
        ```

        设置您的`.env`：

        ```
        OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
        ```

        ### 步骤2：导入和加载API密钥

        ```python
        import os
        from dotenv import load_dotenv
        from langchain_openai import OpenAIEmbeddings

        load_dotenv()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        ```

        ### 步骤3：创建您的语料库

        ```python
        documents = [
            "Artificial Intelligence is the simulation of human intelligence by machines.",
            "Machine learning is a field of AI that uses statistical techniques.",
            "Embeddings convert natural language to numerical vectors.",
            "OpenAI develops powerful language models like GPT-4.",
            "FAISS is a library for efficient similarity search and clustering of dense vectors."
        ]
        ```

        ### 步骤4：生成嵌入

        ```python
        # 使用以下任一方法
        # 简单方法
        embedded_docs = [embeddings.embed_query(doc) for doc in documents]

        # 最优方法
        # embedded_docs = embeddings.embed_documents(documents)
        ```

        ### 步骤5：存储在FAISS中并查询

        ```python
        # 导入FAISS库用于高效的相似性搜索和密集向量聚类
        import faiss

        # 从LangChain社区导入FAISS向量存储实现
        from langchain_community.vectorstores import FAISS

        # 导入内存文档存储，用于与嵌入一起存储文档
        from langchain_community.docstore.in_memory import InMemoryDocstore

        # 嵌入向量的长度
        embedding_dim = len(embeddings.embed_query("hello world"))

        # 为指定嵌入维度创建新的FAISS L2（欧几里得）距离索引
        faiss_index = faiss.IndexFlatL2(embedding_dim)

        # 使用指定的嵌入函数、FAISS索引和内存文档存储创建FAISS向量存储
        # 这将允许对嵌入文档进行高效的相似性搜索
        FAISS_vector_store = FAISS(
            embedding_function=embeddings,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        ```

        现在，嵌入用户查询并执行搜索：

        ```python
        FAISS_vector_store.add_texts(documents)

        # 在FAISS向量存储中执行相似性搜索
        # 这将返回与查询"What is FAISS used for?"最相似的前2个文档
        results = FAISS_vector_store.similarity_search(
            "What is FAISS used for?",
            k=2
        )
        ```

        ### 步骤6：输出示例

        ```
        Best Match: FAISS is a library for efficient similarity search and clustering of dense vectors.
        ```

        用不到50行代码，您就构建了一个功能齐全的语义搜索系统！
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 10. 嵌入的用例和应用

        嵌入不仅仅是一个流行词——它们为现代AI系统中一些最有用的功能提供动力。以下是一些实用的现实世界应用：

        ### 1. 语义搜索

        基于**含义**而非精确关键词的搜索结果。

        **示例：**
        - 查询："How to build neural networks?"
        - 匹配："Guide to deep learning architectures"而不仅仅是精确的关键词匹配。

        ### 2. 推荐系统

        通过比较嵌入向量找到相似的项目、产品或内容。

        **示例：**
        - 查看此文章的用户还喜欢...
        - 电影、书籍或产品相似性

        ### 3. 聊天机器人和RAG应用

        使用向量搜索为大型语言模型（LLM）检索最相关的上下文。

        **示例：**
        - LangChain的检索增强生成（RAG）
        - 从内部文档中提取信息的客户支持机器人

        ### 4. 文本聚类和分类

        在没有明确标签的情况下对相似文本（如评论、工单、推文）进行分组。

        **示例：**
        - 将反馈分组为主题
        - 垃圾邮件vs非垃圾邮件

        ### 5. 异常检测

        使用向量距离识别偏离常态的数据点。

        **示例：**
        - 欺诈检测
        - 异常评论或传感器读数

        ### 6. 个性化

        使用嵌入来个性化用户信息流、内容推荐或搜索结果。

        **示例：**
        - 定制课程推荐
        - 自适应学习平台
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 结论和进一步资源

        在这个旅程中我们走了很长的路——从理解什么是嵌入到使用OpenAI和LangChain构建一个工作的语义搜索系统。

        ### 回顾

        - 嵌入将文本转换为有意义的数值向量。
        - 它们为从搜索引擎和推荐系统到基于LLM的应用的一切提供动力。
        - LangChain和FAISS使生成、存储和高效搜索嵌入变得容易。

        无论您是在构建聊天机器人、智能搜索引擎，还是您的下一个AI副项目——**嵌入都是您的基础**。

        ### 深入学习的资源

        🔗 **博客：LangChain速成课程——第2部分：构建您的第一个RAG应用** [在这里阅读](https://medium.com/@vamshiginna1606/langchain-crash-course-part-2-build-your-first-rag-app-24908b14d337)

        📚 **LangChain文档**: [https://docs.langchain.com](https://docs.langchain.com)

        📖 **OpenAI嵌入API**: [https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)

        🧠 **Meta的FAISS**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

        🆓 **Hugging Face嵌入模型**: [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)

        感谢您的阅读！如果这对您有帮助，请随时分享并在[LinkedIn](https://www.linkedin.com/in/vamshikrishnaginna/)或[Medium](https://medium.com/@vamshiginna1606)上标记我。祝您嵌入愉快！🚀

        ---

        **原文链接：** [Vector Embeddings: From Zero to Hero (with Python & LangChain)](https://medium.com/@vamshiginna1606/vector-embeddings-from-zero-to-hero-with-python-langchain-f5c56e6816cc)
        """
    )
    return


if __name__ == "__main__":
    app.run()
