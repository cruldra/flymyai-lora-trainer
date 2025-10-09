import marimo

__generated_with = "0.15.2"
app = marimo.App(
    width="medium",
    app_title="Tokenization Converting Text to Numbers for Neural Networks",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.8vn9rjdb7m.webp)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 引言：为什么分词很重要

    想象一下，在没有先教会计算机阅读的情况下，试图教它理解莎士比亚的作品。这就是自然语言处理的核心挑战：计算机理解数学，而人类使用文字。分词是连接这两个世界的重要桥梁。

    每当您向ChatGPT提问、在线搜索信息或在电子邮件中获得自动完成建议时，分词都在幕后默默工作，将您的文本转换为驱动这些智能系统的数字序列。

    本文探讨了分词如何将人类语言转换为机器可读的数字，为什么不同的分词方法会极大地影响模型性能，以及如何为您的项目实现生产就绪的分词。无论您是在构建聊天机器人、分析客户反馈，还是训练下一代语言模型，掌握分词对您的成功都至关重要。

    如果您喜欢这篇文章，请在LinkedIn或Medium上关注Rick，获取更多企业AI和AI洞察。

    让我们解码允许机器理解我们的秘密语言。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 第5篇：分词 — 将文本转换为神经网络的数字

    ## 学习目标

    在本教程结束时，您将能够：

    - 理解分词如何将文本转换为数字表示
    - 比较三种主要的分词算法：BPE、WordPiece和Unigram
    - 使用Hugging Face的transformers库实现分词
    - 处理生产系统中的常见边缘情况
    - 有效调试分词问题
    - 为专业领域构建自定义分词器

    这是一个非常实用的文章系列，所以请务必克隆github仓库，运行示例，并加载笔记本。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 引言：为什么分词很重要

    神经网络处理数字，而不是文本。分词将人类语言转换为模型可以理解的数字序列。这种转换决定了您的模型性能如何。

    ### 现实世界的影响

    考虑这些业务场景：

    - **客户支持**：聊天机器人需要区分"can't login"和"cannot log in"
    - **金融分析**：系统必须将"Q4 2023"识别为一个单元，而不是三个
    - **医疗记录**："心肌梗死"必须保持在一起以保持意义

    糟糕的分词会导致：

    - 误解用户意图
    - 错误的数据提取
    - 更高的计算成本
    - 降低模型准确性
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 系统架构概览

    ![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.9rjr701gol.webp)

    架构解释：文本通过分词器，使用预定义的词汇表将其转换为数字ID。这些ID被转换为嵌入向量，然后输入到神经网络中。词汇表在文本片段和数字之间建立映射关系。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 核心概念：文本到Token

    ### 什么是Token？

    Token是模型处理的文本基本单元。它们可以是：

    - **完整单词**：`"cat"` → `["cat"]`
    - **子词**：`"unhappy"` → `["un", "happy"]`
    - **字符**：`"hi"` → `["h", "i"]`

    ### 分词过程

    ![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.102iu7634v.webp)

    过程解释：用户提供文本。分词器在其词汇表中查找每个片段以找到数字ID。特殊token如`[CLS]`和`[SEP]`标记开始和结束。模型接收这些数字进行处理。
    """
    )
    return


@app.cell
def _():
    #这段代码演示了使用BERT进行基础分词：
    import logging

    from transformers import AutoTokenizer

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def demonstrate_basic_tokenization():
        # 展示分词如何将文本转换为数字
        # 这个例子使用BERT的分词器处理一个简单的句子

        # 加载BERT分词器
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # 示例文本
        text = "Tokenization converts text to numbers."

        # 分词
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        # 显示结果
        logger.info(f"原始文本: {text}")
        logger.info(f"Token: {tokens}")
        logger.info(f"Token ID: {token_ids}")

        # 显示token到ID的映射
        for token, token_id in zip(tokens, token_ids[1:-1]):  # 跳过特殊token
            logger.info(f"  '{token}' → {token_id}")

        return tokens, token_ids

    # 运行演示
    tokens, ids = demonstrate_basic_tokenization()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **代码解释**：这个函数加载BERT的分词器并处理一个句子。它显示了文本token和它们的数字ID。映射揭示了每个单词如何被分配相应的数字。特殊token `[CLS]`和`[SEP]`框定序列。

    **函数分析**：`demonstrate_basic_tokenization`

    - **目的**：演示基础的文本到数字转换过程
    - **参数**：无参数
    - **返回值**：元组(tokens: 字符串列表, token_ids: 整数列表)
    - **功能**：将分词结果记录到控制台，首次运行时下载BERT词汇表
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 分词算法

    三种主要算法驱动现代分词。每种算法都在词汇表大小和序列长度之间取得平衡。

    ### 算法比较

    | 算法 | 使用模型 | 方法 | 词汇表大小 | 最适用于 |
    |------|----------|------|------------|----------|
    | **BPE (字节对编码)** | GPT, RoBERTa | 基于频率的合并 | 30k-50k | 通用文本处理 |
    | **WordPiece** | BERT | 似然最大化 | 30k | 多语言应用 |
    | **Unigram** | T5, mBART | 概率模型 | 32k-250k | 灵活的token选择 |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 字节对编码 (BPE)

    BPE通过合并频繁的字符对来构建词汇表：
    """
    )
    return


@app.cell
def _():
    def demonstrate_bpe_tokenization():
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        from transformers import AutoTokenizer
        # 演示使用RoBERTa的BPE分词
        # BPE通过将未知单词分解为已知的子词来处理它们

        tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        # 测试显示BPE行为的单词
        test_words = [
            "tokenization",      # 常见单词
            "pretokenization",   # 复合单词
            "cryptocurrency",    # 技术术语
            "antidisestablish"   # 罕见单词
        ]

        logger.info("=== BPE分词 (RoBERTa) ===")

        for word in test_words:
            tokens = tokenizer.tokenize(word)
            ids = tokenizer.encode(word, add_special_tokens=False)

            logger.info(f"\\n'{word}':")
            logger.info(f"  Token: {tokens}")
            logger.info(f"  数量: {len(tokens)}")

            # 显示BPE如何分割单词
            if len(tokens) > 1:
                logger.info(f"  分割模式: {' + '.join(tokens)}")

        return tokenizer

    # 运行BPE演示
    bpe_tokenizer = demonstrate_bpe_tokenization()
    return


@app.cell
def _(mo):
    mo.md(r"""**代码解释**：BPE分词根据频率将单词分割成更小的单元。常见单词保持完整，而不常见的单词分解为熟悉的部分。这允许处理任何单词，即使是训练数据中未见过的单词。""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## WordPiece分词

    WordPiece使用统计似然来创建子词：
    """
    )
    return


@app.cell
def _():
    def demonstrate_wordpiece_tokenization():
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        from transformers import AutoTokenizer
        # 展示BERT使用的WordPiece分词
        # 注意标记单词延续的##前缀

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # 用于比较的相同测试单词
        test_words = [
            "tokenization",
            "pretokenization",
            "cryptocurrency",
            "antidisestablish"
        ]

        logger.info("\n=== WordPiece分词 (BERT) ===")

        for word in test_words:
            tokens = tokenizer.tokenize(word)

            logger.info(f"\n'{word}':")
            logger.info(f"  Token: {tokens}")

            # 解释##符号
            if any(t.startswith('##') for t in tokens):
                logger.info("  注意: ##表示前一个token的延续")

                # 从片段重构单词
                reconstructed = tokens[0]
                for token in tokens[1:]:
                    reconstructed += token.replace('##', '')
                logger.info(f"  重构: {reconstructed}")

        return tokenizer

    # 运行WordPiece演示
    wordpiece_tokenizer = demonstrate_wordpiece_tokenization()
    return


@app.cell
def _(mo):
    mo.md(r"""**代码解释**：WordPiece用##标记非初始子词。这保留了单词边界，帮助模型理解token关系。重构显示了片段如何重新组合成单词。""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 算法选择指南

    ![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.7lkcl944t3.webp)

    **决策流程**：从您的应用类型开始。通用NLP任务适合使用BPE。多语言应用需要在多样化语言上训练的分词器。技术领域受益于专门的词汇表。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 实现指南

    ### 设置您的环境

    首先，安装所需的依赖：

    ```bash
    # requirements.txt
    transformers==4.36.0
    torch==2.1.0
    tokenizers==0.15.0
    datasets==2.16.0
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 完整的分词管道

    本节演示了一个生产就绪的分词管道：
    """
    )
    return


@app.cell
def _():
    def __():
        import logging

        import torch
        from transformers import AutoTokenizer

        # 配置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        class TokenizationPipeline:
            """
            生产就绪的分词管道，包含错误处理。
            支持批处理和多种输出格式。
            """

            def __init__(self, model_name='bert-base-uncased', max_length=512):
                """
                使用指定模型初始化分词器。

                参数:
                -----------
                model_name : str
                    Hugging Face 模型标识符
                max_length : int
                    最大序列长度
                """
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.max_length = max_length
                logger.info(f"已初始化分词器: {model_name}")

            def tokenize_single(self, text, return_offsets=False):
                """
                对单个文本字符串进行分词。

                参数:
                -----------
                text : str
                    要分词的输入文本
                return_offsets : bool
                    是否返回字符偏移映射

                返回:
                --------
                dict : 分词结果，包括 input_ids, attention_mask
                """
                if not text:
                    logger.warning("提供了空文本")
                    text = ""

                try:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding='max_length',
                        return_offsets_mapping=return_offsets,
                        return_tensors='pt'
                    )

                    logger.info(f"已将 {len(text)} 个字符分词为 {encoding['input_ids'].shape[1]} 个token")
                    return encoding

                except Exception as e:
                    logger.error(f"分词失败: {str(e)}")
                    raise

            def tokenize_batch(self, texts, show_progress=True):
                """
                高效地对一批文本进行分词。

                参数:
                -----------
                texts : list of str
                    要分词的输入文本列表
                show_progress : bool
                    是否显示进度信息

                返回:
                --------
                dict : 批处理分词结果
                """
                if not texts:
                    logger.warning("提供了空文本列表")
                    return None

                # 清理文本
                texts = [text if text else "" for text in texts]

                # 为内存效率分批处理
                batch_size = 32
                all_encodings = []

                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]

                    if show_progress:
                        logger.info(f"正在处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                    encoding = self.tokenizer(
                        batch,
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                        return_tensors='pt'
                    )
                    all_encodings.append(encoding)

                # 合并批次
                combined = {
                    key: torch.cat([e[key] for e in all_encodings], dim=0)
                    for key in all_encodings[0].keys()
                }

                logger.info(f"已分词 {len(texts)} 个文本")
                return combined

        # 创建管道实例
        pipeline = TokenizationPipeline()

        # 演示单文本分词
        logger.info("=== 单文本分词演示 ===")
        sample_text = "This is a sample text for tokenization pipeline demonstration."
        result = pipeline.tokenize_single(sample_text)
        logger.info(f"输入文本: {sample_text}")
        logger.info(f"输出形状: {result['input_ids'].shape}")
        logger.info(f"Token数量: {result['input_ids'].shape[1]}")

        # 演示批处理分词
        logger.info("\n=== 批处理分词演示 ===")
        batch_texts = [
            "First text for batch processing.",
            "Second text with different length.",
            "Third text that is much longer and contains more words for testing purposes."
        ]
        batch_result = pipeline.tokenize_batch(batch_texts)
        logger.info(f"批处理输入: {len(batch_texts)} 个文本")
        logger.info(f"批处理输出形状: {batch_result['input_ids'].shape}")

        # 演示偏移映射
        logger.info("\n=== 偏移映射演示 ===")
        offset_text = "Apple Inc. was founded in 1976."
        offset_result = pipeline.tokenize_single(offset_text, return_offsets=True)
        tokens = pipeline.tokenizer.convert_ids_to_tokens(offset_result['input_ids'][0])
        offsets = offset_result['offset_mapping'][0]

        logger.info(f"原文: {offset_text}")
        logger.info("Token → 原文位置:")
        for token, (start, end) in zip(tokens[:10], offsets[:10]):  # 只显示前10个
            if start == end:
                logger.info(f"  {token:12} → [SPECIAL]")
            else:
                original = offset_text[start:end]
                logger.info(f"  {token:12} → '{original}' [{start}:{end}]")

    __()
    return


@app.cell
def _(mo):
    mo.md(r"""**实现细节**：这个类封装了分词逻辑并提供了适当的错误处理。它支持单个文本和批处理。偏移映射功能使token到字符的对齐成为可能，这对于NER等任务很有用。""")
    return





@app.cell
def _(mo):
    mo.md(
        r"""
    ### 处理特殊Token

    特殊token为序列提供结构：
    """
    )
    return


@app.cell
def _():
    def __():
        import logging

        from transformers import AutoTokenizer

        # 配置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def demonstrate_special_tokens():
            """
            演示特殊token在不同序列类型中的使用。
            展示单序列和序列对中特殊token的位置和作用。
            """
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            # 单个序列
            text1 = "What is tokenization?"
            encoding1 = tokenizer(text1)
            tokens1 = tokenizer.convert_ids_to_tokens(encoding1['input_ids'])

            logger.info("=== 单序列中的特殊Token ===")
            logger.info(f"文本: {text1}")
            logger.info(f"Token: {tokens1}")
            logger.info(f"[CLS] 在位置 0: 标记序列开始")
            logger.info(f"[SEP] 在位置 {len(tokens1)-1}: 标记序列结束")

            # 序列对（用于QA任务）
            question = "What is tokenization?"
            context = "Tokenization converts text into tokens."

            encoding2 = tokenizer(question, context)
            tokens2 = tokenizer.convert_ids_to_tokens(encoding2['input_ids'])

            logger.info("\n=== 序列对中的特殊Token ===")
            logger.info(f"问题: {question}")
            logger.info(f"上下文: {context}")

            # 找到分隔符位置
            sep_positions = [i for i, token in enumerate(tokens2) if token == '[SEP]']
            logger.info(f"[SEP] 位置: {sep_positions}")
            logger.info(f"问题token: 位置 1 到 {sep_positions[0]-1}")
            logger.info(f"上下文token: 位置 {sep_positions[0]+1} 到 {sep_positions[1]-1}")

            return tokens1, tokens2

        # 运行特殊token演示
        tokens1, tokens2 = demonstrate_special_tokens()

    __()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **特殊Token功能**：

    - **[CLS]**: 分类token - 聚合序列含义
    - **[SEP]**: 分隔符token - 标记序列之间的边界
    - **[PAD]**: 填充token - 填充较短序列以匹配批次长度
    - **[UNK]**: 未知token - 替换词汇表外的单词
    - **[MASK]**: 掩码token - 用于掩码语言建模
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 高级功能

    ### NER的偏移映射

    跟踪token在原始文本中的位置：
    """
    )
    return


@app.cell
def _():
    def __():
        import logging

        from transformers import AutoTokenizer

        # 配置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def demonstrate_offset_mapping():
            """
            演示偏移映射功能，用于NER等需要精确字符位置的任务。
            """
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            text = "Apple Inc. was founded by Steve Jobs in Cupertino."
            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True
            )

            tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            offsets = encoding['offset_mapping']

            logger.info("=== Token到字符映射 ===")
            logger.info(f"原文: {text}\n")

            # 创建可视化对齐
            logger.info("Token → 原文 [开始:结束]")
            logger.info("-" * 40)

            for token, (start, end) in zip(tokens, offsets):
                if start == end:  # 特殊token
                    logger.info(f"{token:12} → [SPECIAL]")
                else:
                    original = text[start:end]
                    logger.info(f"{token:12} → '{original}' [{start}:{end}]")

            # 演示实体提取
            entity_tokens = [2, 3]  # "apple inc"
            logger.info(f"\n从token {entity_tokens}提取实体:")

            start_char = offsets[entity_tokens[0]][0]
            end_char = offsets[entity_tokens[-1]][1]
            entity = text[start_char:end_char]
            logger.info(f"提取结果: '{entity}'")

            return encoding

        # 运行偏移映射演示
        encoding = demonstrate_offset_mapping()

    __()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **偏移映射优势**：

    - 保留精确的字符位置
    - 支持在源文本中高亮显示
    - 支持实体提取
    - 在分词过程中保持对齐
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 生产考虑

    ### 性能优化

    分词经常成为瓶颈。以下是优化方法：
    """
    )
    return


@app.cell
def _():
    def __():
        import logging
        import time

        from transformers import AutoTokenizer

        # 配置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def benchmark_tokenization_methods():
            """
            对比不同分词方法的性能。
            测试单独处理、批量处理和快速分词器的速度差异。
            """
            # 创建测试语料库
            texts = ["This is a sample sentence for benchmarking."] * 1000

            # 方法1：单独分词
            tokenizer_slow = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)

            start = time.time()
            for text in texts:
                _ = tokenizer_slow(text)
            individual_time = time.time() - start

            # 方法2：批量分词
            start = time.time()
            _ = tokenizer_slow(texts, padding=True, truncation=True)
            batch_time = time.time() - start

            # 方法3：快速分词器
            tokenizer_fast = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

            start = time.time()
            _ = tokenizer_fast(texts, padding=True, truncation=True)
            fast_time = time.time() - start

            logger.info("=== 性能比较 ===")
            logger.info(f"单独处理: {individual_time:.2f}s")
            logger.info(f"批量处理: {batch_time:.2f}s ({individual_time/batch_time:.1f}x 更快)")
            logger.info(f"快速分词器: {fast_time:.2f}s ({batch_time/fast_time:.1f}x 比批量更快)")

            return {
                'individual': individual_time,
                'batch': batch_time,
                'fast': fast_time
            }

        # 运行性能基准测试
        results = benchmark_tokenization_methods()

    __()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **优化策略**：

    - **使用快速分词器**：基于Rust的实现提供5-10倍加速
    - **批处理**：显著减少开销
    - **尽可能预计算**：缓存分词结果
    - **优化填充**：使用动态填充减少浪费的计算
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 常见问题和解决方案

    ### 问题1：分词器-模型不匹配

    ```python
    def detect_tokenizer_mismatch():
        from transformers import AutoModel

        # 故意不匹配
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('roberta-base')

        text = "This demonstrates tokenizer mismatch."

        try:
            inputs = tokenizer(text, return_tensors='pt')
            outputs = model(**inputs)
            logger.warning("模型处理了不匹配的输入 - 结果不可靠！")
        except Exception as e:
            logger.error(f"不匹配错误: {e}")

        # 正确方法
        logger.info("\\n=== 正确匹配 ===")
        model_name = 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        logger.info(f"成功！输出形状: {outputs.last_hidden_state.shape}")
    ```

    **关键规则**：始终从同一个检查点加载分词器和模型。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 问题2：处理长文档

    ```python
    def handle_long_documents():
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        max_length = 512

        # 创建长文档
        long_doc = " ".join(["This is a sentence."] * 200)

        # 策略1：简单截断
        truncated = tokenizer(
            long_doc,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )

        logger.info(f"文档长度: {len(long_doc)} 字符")
        logger.info(f"截断为: {truncated['input_ids'].shape[1]} token")

        # 策略2：滑动窗口
        stride = 256
        chunks = []

        tokens = tokenizer.tokenize(long_doc)

        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_length - 2]  # 为特殊token保留空间
            chunk_ids = tokenizer.convert_tokens_to_ids(chunk)
            chunk_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
            chunks.append(chunk_ids)

        logger.info(f"\\n滑动窗口创建了 {len(chunks)} 个块")
        logger.info(f"重叠: {max_length - stride} 个token在块之间")

        return chunks
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **长文档策略**：

    - **截断**：快速但会丢失信息
    - **滑动窗口**：保留所有内容但有重叠
    - **分层处理**：分别处理各部分然后合并
    - **摘要**：在分词前减少内容
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 调试分词

    有效的调试可以节省数小时的故障排除时间：

    ```python
    class TokenizationDebugger:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def analyze_text(self, text):
            logger.info(f"\\n=== Analyzing: '{text}' ===")

            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.encode(text)

            logger.info(f"Character count: {len(text)}")
            logger.info(f"Token count: {len(tokens)}")
            logger.info(f"Compression ratio: {len(text)/len(tokens):.2f} chars/token")

            # Check for unknown tokens
            unk_count = tokens.count(self.tokenizer.unk_token)
            if unk_count > 0:
                logger.warning(f"Found {unk_count} unknown tokens!")

            return {
                'tokens': tokens,
                'token_ids': token_ids,
                'char_count': len(text),
                'token_count': len(tokens),
                'unk_count': unk_count
            }
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **调试检查清单**：

    - [ ] 验证分词器与模型匹配
    - [ ] 检查过多的未知token
    - [ ] 监控序列长度
    - [ ] 验证特殊token处理
    - [ ] 测试边缘情况（空字符串、特殊字符）
    - [ ] 与预期输出比较
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 专业领域的自定义分词器

    有时预训练的分词器不适合您的领域。以下是如何创建自定义分词器：

    ```python
    def train_custom_medical_tokenizer():
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        # Medical corpus (in practice, use larger dataset)
        medical_texts = [
            "Patient presents with acute myocardial infarction.",
            "Diagnosis: Type 2 diabetes mellitus with neuropathy.",
            "Prescribed metformin 500mg twice daily.",
            "MRI shows L4-L5 disc herniation with radiculopathy.",
            "Post-operative recovery following cholecystectomy.",
            "Chronic obstructive pulmonary disease exacerbation.",
            "Administered epinephrine for anaphylactic reaction.",
            "ECG reveals atrial fibrillation with rapid ventricular response."
        ]

        # Initialize BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=10000,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2
        )

        # Train on medical corpus
        tokenizer.train_from_iterator(medical_texts, trainer=trainer)

        # Test on medical terms
        test_terms = [
            "myocardial infarction",
            "cholecystectomy",
            "pneumonia",
            "diabetes mellitus"
        ]

        logger.info("=== Custom Medical Tokenizer Results ===")
        for term in test_terms:
            encoding = tokenizer.encode(term)
            logger.info(f"\\n'{term}':")
            logger.info(f"  Tokens: {encoding.tokens}")
            logger.info(f"  IDs: {encoding.ids}")

        return tokenizer
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **自定义分词器优势**：

    - **更好的覆盖率**：保持领域术语完整
    - **更小的词汇表**：专注于相关术语
    - **提高准确性**：更好地表示领域语言
    - **减少Token数**：更高效的处理
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 比较通用分词器与自定义分词器

    ```python
    def compare_medical_tokenization():
        # Generic tokenizer
        generic = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Medical terms that generic tokenizers fragment
        medical_terms = [
            "pneumonoultramicroscopicsilicovolcanoconiosis",
            "electroencephalography",
            "thrombocytopenia",
            "gastroesophageal"
        ]

        logger.info("=== Generic vs Domain Tokenization ===")

        for term in medical_terms:
            generic_tokens = generic.tokenize(term)

            logger.info(f"\\n'{term}':")
            logger.info(f"  Generic: {generic_tokens} ({len(generic_tokens)} tokens)")
            # Custom tokenizer would show fewer tokens

            # Calculate efficiency loss
            if len(generic_tokens) > 3:
                logger.warning(f"  ⚠️ Excessive fragmentation: {len(generic_tokens)} pieces")
    ```

    稍后我们将比较通用分词器与医学术语专用分词器。请阅读到最后，看看它们的比较结果。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 边缘情况和解决方案

    现实世界的文本呈现许多挑战：

    ```python
    def handle_edge_cases():
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        edge_cases = {
            "Empty string": "",
            "Only spaces": "     ",
            "Mixed languages": "Hello 世界 Bonjour",
            "Emojis": "Great job! 👍🎉",
            "Code": "def func(x): return x**2",
            "URLs": "Visit <https://example.com/page>",
            "Special chars": "Price: $99.99 (↑15%)",
            "Long word": "a" * 100
        }

        logger.info("=== Edge Case Handling ===")

        for case_name, text in edge_cases.items():
            logger.info(f"\\n{case_name}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            try:
                tokens = tokenizer.tokenize(text)
                encoding = tokenizer(text, add_special_tokens=True)

                logger.info(f"  Success: {len(tokens)} tokens")

                # Check for issues
                if not tokens and text:
                    logger.warning("  ⚠️ No tokens produced from non-empty text")

                if tokenizer.unk_token in tokens:
                    unk_count = tokens.count(tokenizer.unk_token)
                    logger.warning(f"  ⚠️ Contains {unk_count} unknown tokens")

            except Exception as e:
                logger.error(f"  ❌ Error: {str(e)}")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **常见边缘情况**：

    - **空/空白字符**：返回空token列表或填充token
    - **混合文字**：可能产生未知token
    - **表情符号**：每个分词器处理方式不同
    - **URL/邮箱**：经常被错误分割
    - **超长单词**：可能超过token限制
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 关键要点

    ### 核心概念

    - **分词连接文本和神经网络** — 这是决定模型性能的关键第一步
    - **算法选择很重要** — BPE、WordPiece和Unigram各自在不同应用中有优势
    - **始终匹配分词器和模型** — 不匹配会导致静默失败和糟糕结果
    - **特殊token提供结构** — [CLS]、[SEP]等帮助模型理解序列
    - **生产需要优化** — 使用快速分词器和批处理提高效率
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 最佳实践检查清单

    - [ ] 训练和推理使用相同的分词器
    - [ ] 优雅处理边缘情况（空字符串、特殊字符）
    - [ ] 实现适当的错误处理和日志记录
    - [ ] 针对生产约束优化（速度vs准确性）
    - [ ] 使用真实世界数据测试，包括边缘情况
    - [ ] 监控分词指标（未知token率、序列长度）
    - [ ] 考虑专业应用的领域特定分词器
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 快速参考

    ```python
    # 标准设置
    from transformers import AutoTokenizer

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # 基础用法
    tokens = tokenizer.tokenize("Hello world")
    encoding = tokenizer("Hello world", return_tensors='pt')

    # 生产用法
    encoding = tokenizer(
        texts,                    # 字符串列表
        padding=True,            # 填充到相同长度
        truncation=True,         # 截断到max_length
        max_length=512,         # 最大序列长度
        return_tensors='pt',    # 返回PyTorch张量
        return_attention_mask=True,  # 返回注意力掩码
        return_offsets_mapping=True  # 用于NER任务
    )

    # 访问结果
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 下一步

    1. **在您的数据上实验不同的分词器**
    2. **测量您用例的分词指标**
    3. **如需要构建自定义分词器**
    4. **与您的模型管道集成**
    5. **监控生产性能**

    分词可能看起来简单，但它是每个NLP系统的基础。掌握它，您将构建更强大和高效的应用程序。

    现在，让我们实际使用这些示例。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 系列文章

    如果您喜欢这篇文章，请在LinkedIn或Medium上关注Rick，获取更多企业AI和AI洞察。

    请务必查看本系列的前四篇文章：

    1. [**Hugging Face Transformers和AI革命**（第1篇）](https://medium.com/@richardhightower/transformers-and-the-ai-revolution-the-role-of-hugging-face-f185f574b91b)
    2. [**Hugging Face：为什么语言对AI来说很困难？Transformer如何改变这一点**（第2篇）](https://medium.com/@richardhightower/why-language-is-hard-for-ai-and-how-transformers-changed-everything-d8a1fa299f1e)
    3. [**Hugging Face实践：构建您的AI工作空间**（第3篇）](https://medium.com/@richardhightower/hands-on-with-hugging-face-building-your-ai-workspace-b23c7e9be3a7)
    4. [**Transformer内部：架构和注意力机制揭秘**（第4篇）](https://medium.com/@richardhightower/inside-the-transformer-architecture-and-attention-demystified-39b2c13130bd)
    5. **分词 — 将文本转换为神经网络的数字**（第5篇 - 本篇）
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## GitHub仓库使用说明

    ### 分词 — 将文本转换为神经网络的数字

    本项目包含Hugging Face Transformers系列第5篇文章：分词的工作示例。

    🔗 **GitHub仓库**: https://github.com/RichardHightower/art_hug_05

    ### 前置要求

    - Python 3.12（通过pyenv管理）
    - Poetry用于依赖管理
    - Go Task用于构建自动化
    - 所需服务的API密钥（参见.env.example）
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 设置步骤

    1. **克隆仓库**：
    ```bash
    git clone git@github.com:RichardHightower/art_hug_05.git
    cd art_hug_05
    ```

    2. **运行设置任务**：
    ```bash
    task setup
    ```

    3. **配置环境**：
    ```bash
    cp .env.example .env
    # 根据需要配置.env文件
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 项目结构

    ```
    .
    ├── src/
    │   ├── __init__.py
    │   ├── config.py              # 配置和工具
    │   ├── main.py                # 包含所有示例的入口点
    │   ├── tokenization_examples.py       # 基础分词示例
    │   ├── tokenization_algorithms.py     # BPE、WordPiece和Unigram比较
    │   ├── custom_tokenization.py         # 训练自定义分词器
    │   ├── tokenization_debugging.py      # 调试和可视化工具
    │   ├── multimodal_tokenization.py     # 图像和CLIP分词
    │   ├── advanced_tokenization.py       # 高级分词技术
    │   ├── model_loading.py               # 模型加载示例
    │   └── utils.py               # 工具函数
    ├── tests/
    │   └── test_examples.py       # 单元测试
    ├── .env.example               # 环境模板
    ├── Taskfile.yml               # 任务自动化
    └── pyproject.toml             # Poetry配置
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 运行示例

    **运行所有示例**：
    ```bash
    task run
    ```

    **或运行单个模块**：
    ```bash
    task run-tokenization          # 运行基础分词示例
    task run-algorithms            # 运行分词算法比较
    task run-custom                # 运行自定义分词器训练
    task run-debugging             # 运行分词调试工具
    task run-multimodal            # 运行多模态分词
    task run-advanced              # 运行高级分词技术
    task run-model-loading         # 运行模型加载示例
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 加载笔记本

    启动Jupyter笔记本：
    ```bash
    task notebook
    ```

    这将启动一个Jupyter服务器，您可以：

    - 创建交互式笔记本进行实验
    - 逐步运行代码单元
    - 可视化分词结果
    - 交互式测试不同的分词器
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 可用任务

    - `task setup` - 设置Python环境并安装依赖
    - `task run` - 运行所有示例
    - `task run-tokenization` - 运行基础分词示例
    - `task run-algorithms` - 运行算法比较示例
    - `task run-custom` - 运行自定义分词器训练
    - `task run-debugging` - 运行调试和可视化工具
    - `task run-multimodal` - 运行多模态分词示例
    - `task run-advanced` - 运行高级分词技术
    - `task run-model-loading` - 运行模型加载示例
    - `task notebook` - 启动Jupyter笔记本服务器
    - `task test` - 运行单元测试
    - `task format` - 使用Black和Ruff格式化代码
    - `task lint` - 运行代码检查（Black、Ruff、mypy）
    - `task clean` - 清理生成的文件
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 在Mac和Windows上设置Python和Go Task

    ### 安装Python

    #### 在macOS上

    1. **使用Homebrew（推荐）**：
    ```bash
    brew install pyenv
    ```

    2. **使用pyenv安装Python 3.12**：
    ```bash
    pyenv install 3.12.0
    pyenv global 3.12.0
    ```

    3. **验证安装**：
    ```bash
    python --version
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### 在Windows上

    1. **从Python.org下载安装程序**
    2. **运行安装程序并确保勾选"Add Python to PATH"**
    3. **打开命令提示符并验证安装**：
    ```cmd
    python --version
    ```

    4. **安装pyenv for Windows（可选）**：
    ```cmd
    pip install pyenv-win
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 安装Poetry

    #### 在macOS上

    1. **使用官方安装程序**：
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    2. **将Poetry添加到PATH**：
    ```bash
    echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### 在Windows上

    1. **使用PowerShell安装**：
    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```

    2. **将Poetry添加到PATH（安装程序应该自动完成）**

    3. **验证安装**：
    ```cmd
    poetry --version
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 安装Go Task

    #### 在macOS上

    1. **使用Homebrew**：
    ```bash
    brew install go-task/tap/go-task
    ```

    2. **验证安装**：
    ```bash
    task --version
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### 在Windows上

    1. **使用Scoop**：
    ```cmd
    scoop install go-task
    ```

    2. **或使用Chocolatey**：
    ```cmd
    choco install go-task
    ```

    3. **或从GitHub Releases直接下载并添加到PATH**

    4. **验证安装**：
    ```cmd
    task --version
    ```

    ---

    现在您已经拥有了完整的分词知识和实践工具！开始探索和实验吧！
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 项目设置

    安装所有前置要求后，您可以按照前面部分的设置说明来运行项目。

    ### 常见问题排除

    - **Python未找到**：确保Python正确添加到PATH变量
    - **Poetry命令不工作**：重启终端或将Poetry bin目录添加到PATH
    - **Task未找到**：验证Task安装和PATH设置
    - **依赖错误**：运行`poetry update`解决依赖冲突
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 医学分词示例

    我们创建了一个比较专业医学分词与非医学分词的示例。

    ```bash
    task run-medical
    ```

    **输出示例**：
    ```
    INFO:__main__:🏥 Medical Tokenization Examples
    INFO:__main__:==================================================
    INFO:__main__:
    === Generic vs Domain Tokenization ===
    INFO:__main__:
    'pneumonoultramicroscopicsilicovolcanoconiosis':
    INFO:__main__:  Generic: ['p', '##ne', '##um', '##ono', '##ult', '##ram', '##ic', '##ros', '##copic', '##sil', '##ico', '##vo', '##lc', '##ano', '##con', '##ios', '##is'] (17 tokens)
    WARNING:__main__:  ⚠️ Excessive fragmentation: 17 pieces

    'electroencephalography':
    INFO:__main__:  Generic: ['electro', '##ence', '##pha', '##log', '##raphy'] (5 tokens)
    WARNING:__main__:  ⚠️ Excessive fragmentation: 5 pieces
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### MedCPT vs 通用BERT比较结果

    ```
    === Comparison with Generic BERT ===

    'diabetes insipidus':
      MedCPT: 4 tokens
      Generic BERT: 5 tokens
      ✅ MedCPT is 1 tokens more efficient

    'vasopressinergic neurons':
      MedCPT: 3 tokens
      Generic BERT: 6 tokens
      ✅ MedCPT is 3 tokens more efficient

    'hypothalamic destruction':
      MedCPT: 2 tokens
      Generic BERT: 6 tokens
      ✅ MedCPT is 4 tokens more efficient

    'polyuria and polydipsia':
      MedCPT: 6 tokens
      Generic BERT: 7 tokens
      ✅ MedCPT is 1 tokens more efficient
    ```

    可以看到专业模型在医学术语方面比通用模型更高效。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 医学分词演示代码

    让我们检查驱动医学分词演示的代码。下面的脚本比较了专业医学分词器如何处理复杂医学术语与通用分词器的对比：

    ```python
    # Medical Tokenization Demo
    # Standalone script to run medical tokenization examples

    from transformers import AutoTokenizer, AutoModel
    import torch
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    def compare_medical_tokenization():
        # Generic tokenizer
        generic = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Medical terms that generic tokenizers fragment
        medical_terms = [
            "pneumonoultramicroscopicsilicovolcanoconiosis",
            "electroencephalography",
            "thrombocytopenia",
            "gastroesophageal"
        ]

        logger.info("\\n=== Generic vs Domain Tokenization ===")

        for term in medical_terms:
            generic_tokens = generic.tokenize(term)

            logger.info(f"\\n'{term}':")
            logger.info(f"  Generic: {generic_tokens} ({len(generic_tokens)} tokens)")

            # Calculate efficiency loss
            if len(generic_tokens) > 3:
                logger.warning(f"  ⚠️ Excessive fragmentation: {len(generic_tokens)} pieces")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ```python
    def medcpt_encoder_example():
        logger.info("\\n=== MedCPT Biomedical Text Encoder Example ===")

        try:
            # Load MedCPT Article Encoder
            logger.info("Loading MedCPT Article Encoder...")
            model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
            tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

            # Example medical articles
            articles = [
                [
                    "Diagnosis and Management of Central Diabetes Insipidus in Adults",
                    "Central diabetes insipidus (CDI) is a clinical syndrome...",
                ],
                [
                    "Adipsic diabetes insipidus",
                    "Adipsic diabetes insipidus (ADI) is a rare but devastating disorder...",
                ],
                [
                    "Nephrogenic diabetes insipidus: a comprehensive overview",
                    "Nephrogenic diabetes insipidus (NDI) is characterized by...",
                ],
            ]

            # Format articles for the model
            formatted_articles = [f"{title}. {abstract}" for title, abstract in articles]

            with torch.no_grad():
                # Tokenize the articles
                encoded = tokenizer(
                    formatted_articles,
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                    max_length=512,
                )

                # Encode the articles
                embeds = model(**encoded).last_hidden_state[:, 0, :]

                logger.info(f"\\nEmbedding shape: {embeds.shape}")
                logger.info(f"Embedding dimension: {embeds.shape[1]}")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ```python
                # Show tokenization comparison for medical terms
                logger.info("\\n=== MedCPT Tokenization of Medical Terms ===")

                medical_terms = [
                    "diabetes insipidus",
                    "vasopressinergic neurons",
                    "hypothalamic destruction",
                    "polyuria and polydipsia"
                ]

                for term in medical_terms:
                    tokens = tokenizer.tokenize(term)
                    logger.info(f"\\n'{term}':")
                    logger.info(f"  Tokens: {tokens} ({len(tokens)} tokens)")

                # Compare with generic BERT tokenizer
                generic = AutoTokenizer.from_pretrained('bert-base-uncased')
                logger.info("\\n=== Comparison with Generic BERT ===")

                for term in medical_terms:
                    medcpt_tokens = tokenizer.tokenize(term)
                    generic_tokens = generic.tokenize(term)

                    logger.info(f"\\n'{term}':")
                    logger.info(f"  MedCPT: {len(medcpt_tokens)} tokens")
                    logger.info(f"  Generic BERT: {len(generic_tokens)} tokens")

                    if len(generic_tokens) > len(medcpt_tokens):
                        logger.info(f"  ✅ MedCPT is {len(generic_tokens) - len(medcpt_tokens)} tokens more efficient")

        except Exception as e:
            logger.error(f"Error loading MedCPT model: {e}")
            logger.info("Install with: pip install transformers torch")
            logger.info("Note: MedCPT model requires downloading ~440MB")


    def main():
        logger.info("🏥 Medical Tokenization Examples")
        logger.info("=" * 50)

        # Run generic vs domain comparison
        compare_medical_tokenization()

        # Run MedCPT encoder example
        medcpt_encoder_example()

        logger.info("\\n✅ Medical tokenization examples completed!")


    if __name__ == "__main__":
        main()
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 示例分析

    这个示例演示了专业医学分词与通用分词的工作原理对比。让我们分解一下：

    **示例包含三个主要部分**：

    1. **通用vs领域分词比较**：显示标准分词器如何将复杂医学术语分解为许多小片段（token）
    2. **MedCPT编码器示例**：演示专门的医学文本编码器模型，更好地理解医学术语
    3. **分词器之间的比较**：直接比较使用两种分词器处理相同医学短语需要多少token

    **结果清楚显示**：

    - 通用分词器在医学术语方面表现困难
    - 例如，它们将"hypothalamic destruction"分割为6个token，而医学分词器只需要2个token
    - 更少的token意味着更高效的处理（节省时间和计算资源）
    - 更好的分词导致更好的文本含义理解
    - 专业模型可以在token限制内处理更长的医学文本

    **示例加载两种不同的分词器**：

    - 通用的"bert-base-uncased"，适用于日常语言
    - 专门的"MedCPT-Article-Encoder"，专门在医学文本上训练

    结果确认了文章讨论的内容：领域特定分词对专业文本显著更高效，在某些情况下减少高达66%的token数量，直接影响模型性能和成本。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 笔记本示例

    有几个笔记本可以让您浏览本文中的大部分示例。只需从上述仓库下载源代码，然后运行`task notebook`，导航到notebooks文件夹并加载笔记本，运行示例。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.7zqsc57zfj.webp)

    ![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.8dx830geda.webp)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 关于作者

    Rick Hightower拥有丰富的企业经验，曾担任财富100强公司的高管和杰出工程师，专门从事机器学习和AI解决方案，为客户提供智能体验。他的专业知识涵盖AI技术的理论基础和实际应用。

    作为TensorFlow认证专业人士和斯坦福大学机器学习专业化课程的毕业生，Rick将学术严谨性与实际实施经验相结合。他的培训包括掌握监督学习技术、神经网络和高级AI概念，并已成功将这些应用于企业级解决方案。

    凭借对AI实施的业务和技术方面的深入理解，Rick在理论机器学习概念和实际业务应用之间架起了桥梁，帮助组织利用AI创造有形价值。

    在LinkedIn或Medium上关注Rick，获取更多企业AI和AI洞察。
    """
    )
    return


if __name__ == "__main__":
    app.run()
