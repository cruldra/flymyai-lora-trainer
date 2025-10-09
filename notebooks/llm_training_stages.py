import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 从零开始训练LLM的4个阶段

    今天，我们将介绍从零开始构建LLM的4个阶段，这些阶段使LLM能够适用于现实世界的用例。

    我们将涵盖：

    - 预训练（Pre-training）
    - 指令微调（Instruction fine-tuning）
    - 偏好微调（Preference fine-tuning）
    - 推理微调（Reasoning fine-tuning）

    下面的视觉图总结了这些技术：

    ![LLM训练四个阶段](https://substackcdn.com/image/fetch/$s_!es-L!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa2adad94-17f7-43e0-b23f-81626248cf0b_922x1096.gif)

    让我们深入了解！

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 0️⃣ 随机初始化的LLM

    在这个阶段，模型什么都不知道。

    你问它"什么是LLM？"，它会给出像"try peter hand and hello 448Sn"这样的胡言乱语。

    它还没有看到任何数据，只拥有随机权重。

    ![随机初始化模型](https://substackcdn.com/image/fetch/$s_!nNqU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb9eac33-37a4-414f-bf9d-f5a637c9dfa4_1000x356.gif)

    在这个初始状态下，模型：

    - 权重完全随机
    - 没有任何语言知识
    - 输出完全不可预测
    - 无法理解任何输入
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1️⃣ 预训练（Pre-training）

    这个阶段通过在大规模语料库上训练模型来预测下一个token，从而教会LLM语言的基础知识。通过这种方式，它吸收了语法、世界事实等。

    但它不擅长对话，因为当被提示时，它只是继续文本。

    ![预训练阶段](https://substackcdn.com/image/fetch/$s_!H24w!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7c2b8c79-de81-4c65-9f81-96277f53de25_1474x568.gif)

    ### 预训练的关键特点：

    - **训练目标**：下一个token预测
    - **数据规模**：数万亿个token的文本数据
    - **学习内容**：
        - 语法规则
        - 世界知识
        - 语言模式
        - 常识推理
    - **局限性**：只会继续文本，不会对话

    ### 技术实现要点：

    [我们在这里从零实现了Llama 4的预训练 →](https://www.dailydoseofds.com/building-llama-4-from-scratch-with-python/)

    涵盖内容：

    - 字符级分词
    - 带旋转位置编码（RoPE）的多头自注意力
    - 多专家MLP的稀疏路由
    - RMSNorm、残差连接和因果掩码
    - 最后是训练和生成
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2️⃣ 指令微调（Instruction Fine-tuning）

    为了使其具有对话能力，我们通过在指令-响应对上训练来进行指令微调。这帮助它学会如何遵循提示并格式化回复。

    ![指令微调](https://substackcdn.com/image/fetch/$s_!-YbP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbc142d91-70e2-433c-bd58-d3460dac7f75_1300x536.gif)

    ### 指令微调后，模型现在可以：

    - 回答问题
    - 总结内容
    - 编写代码
    - 遵循复杂指令
    - 进行对话交互

    ### 训练数据特点：

    - **格式**：指令-响应对
    - **质量**：人工标注的高质量数据
    - **多样性**：涵盖各种任务类型
    - **规模**：通常几十万到几百万个样本

    ### 面临的挑战：

    在这个阶段，我们可能已经：

    - 利用了整个原始互联网档案和知识
    - 用完了人工标注指令响应数据的预算

    那么我们能做什么来进一步改进模型呢？

    我们进入了**强化学习（RL）**的领域。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3️⃣ 偏好微调（Preference Fine-tuning, PFT）

    你一定在ChatGPT上看过这个界面，它问：你更喜欢哪个回答？

    ![ChatGPT偏好选择](https://substackcdn.com/image/fetch/$s_!grMf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F578cc71c-3ca4-4e49-8a25-428f48c7f11a_680x570.png)

    这不仅仅是为了反馈，而是有价值的人类偏好数据。

    OpenAI使用这些数据通过偏好微调来微调他们的模型。

    ### PFT的工作流程：

    用户在2个响应之间选择，产生人类偏好数据。

    然后训练一个奖励模型来预测人类偏好，并使用强化学习更新LLM。

    ![偏好微调流程](https://substackcdn.com/image/fetch/$s_!h1yO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F62c8db89-d9e0-44d1-b1bb-cdfa4c2d474d_1438x486.gif)

    ### 技术细节：

    - **方法名称**：RLHF（人类反馈强化学习）
    - **算法**：PPO（近端策略优化）
    - **目标**：使LLM与人类价值观对齐
    - **适用场景**：没有"正确"答案的开放性问题

    ### 关键优势：

    它教会LLM在没有"正确"答案的情况下与人类保持一致。

    但我们还可以进一步改进LLM。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4️⃣ 推理微调（Reasoning Fine-tuning）

    在推理任务（数学、逻辑等）中，通常只有一个正确的响应和获得答案的明确步骤序列。

    所以我们不需要人类偏好，可以使用正确性作为信号。

    这被称为推理微调。

    ![推理微调](https://substackcdn.com/image/fetch/$s_!FhVP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F13fabee7-4419-46b0-ac74-fdd699a72a3e_1282x422.gif)

    ### 推理微调的步骤：

    1. 模型对提示生成答案
    2. 将答案与已知的正确答案进行比较
    3. 基于正确性分配奖励

    ### 技术特点：

    - **方法名称**：RLVR（可验证奖励强化学习）
    - **流行技术**：DeepSeek的GRPO
    - **奖励信号**：答案的正确性
    - **适用领域**：数学、逻辑、编程等有明确答案的任务

    ### 与偏好微调的区别：

    | 特征 | 偏好微调 | 推理微调 |
    |------|----------|----------|
    | 奖励来源 | 人类偏好 | 答案正确性 |
    | 适用任务 | 开放性问题 | 有明确答案的任务 |
    | 评估标准 | 主观偏好 | 客观正确性 |
    | 数据需求 | 人工标注 | 自动验证 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 总结：LLM训练的完整流程

    这就是训练LLM的4个阶段：

    ### 🔄 完整训练流程：

    1. **从随机初始化模型开始**
           - 权重完全随机
           - 输出毫无意义

    3. **在大规模语料库上进行预训练**
           - 学习语言基础
           - 获得世界知识
           - 掌握语法规则

    5. **使用指令微调使其遵循命令**
           - 学会对话交互
           - 理解任务指令
           - 格式化输出

    7. **使用偏好和推理微调来优化响应**
           - 偏好微调：与人类价值观对齐
           - 推理微调：提高逻辑推理能力

    ### 🎯 每个阶段的核心目标：

    | 阶段 | 主要目标 | 训练方法 | 数据类型 |
    |------|----------|----------|----------|
    | 预训练 | 语言理解 | 下一个token预测 | 大规模文本 |
    | 指令微调 | 任务执行 | 监督学习 | 指令-响应对 |
    | 偏好微调 | 价值对齐 | 强化学习 | 人类偏好 |
    | 推理微调 | 逻辑推理 | 强化学习 | 正确性验证 |

    ### 🚀 未来展望：

    在未来的文章中，我们将深入探讨这些技术的具体实现。

    **同时，请阅读我们从零实现Llama 4预训练的文章 →** [链接](https://www.dailydoseofds.com/building-llama-4-from-scratch-with-python/)

    涵盖内容：

    - 字符级分词
    - 带旋转位置编码（RoPE）的多头自注意力
    - 多专家MLP的稀疏路由
    - RMSNorm、残差连接和因果掩码
    - 最后是训练和生成

    通过理解这四个阶段，你现在对现代LLM的训练流程有了全面的认识！
    """
    )
    return


if __name__ == "__main__":
    app.run()
