import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 使用 GRPO 构建推理 LLM

    本笔记本介绍如何使用 Group Relative Policy Optimization (GRPO) 方法从零开始构建一个推理大语言模型。

    **原文链接**: [Build a Reasoning LLM using GRPO](https://lightning.ai/lightning-purchase-test/studios/build-a-reasoning-llm-from-scratch-using-grpo)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 什么是 GRPO？

    Group Relative Policy Optimization 是一种强化学习方法，它使用确定性奖励函数对 LLM 进行数学和推理任务的微调，无需标注数据。

    ### GRPO 工作流程：

    1. **准备数据集** - 添加推理导向的系统提示（例如："一步步思考..."）
    2. **生成候选响应** - LLM 使用采样引擎生成多个候选响应
    3. **分配奖励** - 每个响应被分配奖励，汇总后为每个生成的响应产生一个分数
    4. **优化更新** - GRPO 损失函数使用这些奖励计算梯度，通过反向传播更新 LLM，模型随时间提高推理能力
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 技术栈

    本教程使用以下工具：

    - **UnslothAI** - 用于高效微调
    - **HuggingFace TRL** - 应用 GRPO 算法
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 步骤 1: 加载模型

    我们使用 Unsloth 加载 Qwen3-4B-Base 模型及其分词器。

    你可以在这里使用任何其他开源权重的 LLM。
    """
    )
    return


@app.cell
def _():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Base",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    return FastLanguageModel, model, tokenizer


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 步骤 2: 定义 LoRA 配置

    我们使用 LoRA 来避免微调整个模型权重。在这段代码中，我们通过指定以下内容使用 Unsloth 的 PEFT：

    - 模型
    - LoRA 低秩 (r)
    - 微调的模块等
    """
    )
    return


@app.cell
def _(FastLanguageModel):
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return (model,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 步骤 3: 创建数据集

    我们加载 Open R1 Math 数据集（一个数学问题数据集）并将其格式化用于推理。

    每个样本包括：

    - 强制结构化推理的系统提示
    - 来自数据集的问题
    - 所需格式的答案
    """
    )
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("openr1/open-r1-math", split="train")

    def format_prompt(sample):
        system_prompt = """You are a helpful assistant. Think step by step and provide detailed reasoning before giving your final answer.
    Format your response as:
    <reasoning>
    [Your step-by-step thinking here]
    </reasoning>
    <answer>
    [Your final answer here]
    </answer>"""

        return {
            "prompt": f"{system_prompt}\n\nQuestion: {sample['question']}",
            "answer": sample["answer"],
        }

    formatted_dataset = dataset.map(format_prompt)
    return (formatted_dataset,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 步骤 4: 定义奖励函数

    在 GRPO 中，我们使用确定性函数来验证响应并分配奖励。无需人工标注！

    奖励函数包括：

    - 精确匹配格式
    - 近似匹配格式
    - 检查答案
    - 检查数字
    """
    )
    return


@app.cell
def _():
    import re

    def exact_format_match(response):
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        return 1.0 if re.search(pattern, response, re.DOTALL) else 0.0

    def approximate_format_match(response):
        has_reasoning = "<reasoning>" in response.lower()
        has_answer = "<answer>" in response.lower()
        return 0.5 if (has_reasoning or has_answer) else 0.0

    def check_answer(response, ground_truth):
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            predicted = answer_match.group(1).strip()
            return 2.0 if predicted == ground_truth else 0.0
        return 0.0

    def check_numbers(response, ground_truth):
        response_numbers = set(re.findall(r"\d+\.?\d*", response))
        truth_numbers = set(re.findall(r"\d+\.?\d*", ground_truth))
        if truth_numbers:
            overlap = len(response_numbers & truth_numbers) / len(truth_numbers)
            return overlap
        return 0.0

    def compute_reward(response, ground_truth):
        reward = 0.0
        reward += exact_format_match(response)
        reward += approximate_format_match(response)
        reward += check_answer(response, ground_truth)
        reward += check_numbers(response, ground_truth)
        return reward
    return (compute_reward,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 步骤 5: 使用 GRPO 开始训练

    现在我们已经准备好数据集和奖励函数，是时候应用 GRPO 了。

    HuggingFace TRL 以 GRPOConfig 和 GRPOTrainer 的形式提供了我们在 GRPO 图表中描述的所有内容。
    """
    )
    return


@app.cell
def _(compute_reward, formatted_dataset, model, tokenizer):
    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir="./grpo_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        num_generation_samples=4,
        max_new_tokens=512,
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        reward_function=lambda responses, prompts: [
            compute_reward(r, p) for r, p in zip(responses, prompts)
        ],
    )
    return (trainer,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 开始训练

    运行以下单元格开始训练过程：
    """
    )
    return


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 对比结果

    我们可以看到 GRPO 如何将基础模型转变为推理强大的模型。

    训练前后的对比显示，模型在以下方面有显著改进：

    - 结构化推理能力
    - 答案准确性
    - 格式遵循能力
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 何时使用强化微调 (RFT) vs 监督微调 (SFT)？

    ### 使用 RFT (如 GRPO) 当：

    - 你有明确的奖励信号（如正确答案、代码执行结果）
    - 任务需要探索和优化（如推理、数学问题）
    - 你想要模型学习策略而不仅仅是模仿

    ### 使用 SFT 当：

    - 你有高质量的标注数据
    - 任务是模仿特定风格或格式
    - 你需要快速适应特定领域
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 总结

    GRPO 是一种强大的方法，可以在没有标注数据或人工干预的情况下，将任何模型转变为推理强大的模型。

    关键要点：

    1. **无需标注数据** - 使用确定性奖励函数
    2. **高效训练** - 结合 LoRA 和 Unsloth
    3. **灵活应用** - 可用于各种推理任务

    **完整代码**: [Build a reasoning LLM from scratch using GRPO](https://lightning.ai/lightning-purchase-test/studios/build-a-reasoning-llm-from-scratch-using-grpo)
    """
    )
    return


if __name__ == "__main__":
    app.run()
