"""
TRL 完全指南 - Transformer Reinforcement Learning

TRL是Hugging Face开发的全栈库，用于训练transformer语言模型，
支持监督微调(SFT)、强化学习(PPO/RLOO)、直接偏好优化(DPO)等方法。

特点：
1. 🚀 全栈解决方案 - 从SFT到RLHF的完整工具链
2. 🎯 多种训练方法 - SFT、PPO、DPO、KTO、ORPO等
3. 🔧 易于集成 - 与Transformers、PEFT无缝集成
4. 📊 奖励建模 - 内置奖励模型训练支持
5. 🌐 分布式训练 - 支持DeepSpeed、FSDP等

作者: Marimo Notebook
日期: 2025-01-XX
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="TRL 完全指南")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🎨 TRL 完全指南

    ## 什么是TRL？

    **TRL (Transformer Reinforcement Learning)** 是Hugging Face开发的全栈库，专门用于训练和微调transformer语言模型。它提供了从监督微调到强化学习的完整工具链。

    ### 核心特性

    1. **监督微调 (SFT)** 📝
       - 基础模型微调
       - 指令微调
       - 对话模型训练

    2. **强化学习 (RL)** 🎮
       - PPO (Proximal Policy Optimization)
       - RLOO (Reinforce Leave One Out)
       - GRPO (Group Relative Policy Optimization)

    3. **偏好优化** 🎯
       - DPO (Direct Preference Optimization)
       - KTO (Kahneman-Tversky Optimization)
       - ORPO (Odds Ratio Preference Optimization)
       - CPO (Contrastive Preference Optimization)

    4. **奖励建模** 🏆
       - 奖励模型训练
       - 过程奖励模型 (PRM)
       - 自定义奖励函数

    5. **高级功能** ⚡
       - 分布式训练支持
       - 内存优化
       - vLLM集成
       - PEFT集成

    ### TRL vs 其他训练框架

    | 特性 | TRL | Axolotl | LLaMA-Factory | DeepSpeed-Chat |
    |------|-----|---------|---------------|----------------|
    | SFT支持 | ✅ | ✅ | ✅ | ✅ |
    | RLHF支持 | ✅ 完整 | ⭐⭐ 基础 | ⭐⭐ 基础 | ✅ 完整 |
    | DPO支持 | ✅ | ✅ | ✅ | ❌ |
    | 易用性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
    | 灵活性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
    | 官方支持 | ✅ HF官方 | 社区 | 社区 | ✅ MS官方 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📦 安装和环境配置

    ### 方式1: 基础安装

    ```bash
    # 基础安装
    pip install trl

    # 或使用uv
    uv pip install trl
    ```

    ### 方式2: 完整安装（包含所有依赖）

    ```bash
    # 安装所有可选依赖
    pip install trl[all]

    # 或分别安装特定功能
    pip install trl[peft]      # PEFT集成
    pip install trl[deepspeed] # DeepSpeed支持
    pip install trl[diffusers] # 图像生成模型支持
    ```

    ### 方式3: 从源码安装（开发版本）

    ```bash
    git clone https://github.com/huggingface/trl.git
    cd trl
    pip install -e .
    ```

    ### 依赖要求

    - Python >= 3.8
    - PyTorch >= 2.0
    - Transformers >= 4.36
    - Accelerate >= 0.20
    - Datasets >= 2.0
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1️⃣ 监督微调 (SFT)

    ### 什么是SFT？

    监督微调是使用标注数据对预训练模型进行微调的过程。这是训练指令模型的第一步。

    ### SFT的应用场景

    - 指令跟随模型
    - 对话模型
    - 特定任务微调
    - 领域适应

    ### 数据格式

    TRL支持多种数据格式：

    | 格式 | 说明 | 示例 |
    |------|------|------|
    | 标准格式 | `{"text": "..."}` | 单轮对话 |
    | 对话格式 | `{"messages": [...]}` | 多轮对话 |
    | 指令格式 | `{"prompt": "...", "completion": "..."}` | 指令-响应对 |
    """
    )
    return


@app.cell
def _():
    print("=" * 60)
    print("📝 SFT训练示例")
    print("=" * 60)

    # 准备示例数据
    sft_example_data = [
        {
            "text": "<|user|>\n你好，请介绍一下自己。<|assistant|>\n你好！我是一个AI助手，很高兴为你服务。"
        },
        {
            "text": "<|user|>\n什么是机器学习？<|assistant|>\n机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能。"
        },
        {
            "text": "<|user|>\n如何学习Python？<|assistant|>\n学习Python可以从基础语法开始，然后通过实践项目来巩固知识。"
        }
    ]

    print(f"\n✅ 准备了 {len(sft_example_data)} 条训练数据")
    print("\n示例数据:")
    for sft_idx, sft_item in enumerate(sft_example_data[:2], 1):
        print(f"\n{sft_idx}. {sft_item['text'][:80]}...")

    # SFT训练配置示例
    sft_config_example = """
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# 2. 配置训练参数
config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_seq_length=512,
    logging_steps=10,
    save_steps=100,
)

# 3. 创建训练器
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 4. 开始训练
trainer.train()
"""

    print("\n" + "=" * 60)
    print("📋 SFT训练配置示例:")
    print("=" * 60)
    print(sft_config_example)

    return sft_config_example, sft_example_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2️⃣ 直接偏好优化 (DPO)

    ### 什么是DPO？

    DPO是一种无需奖励模型的偏好优化方法，直接从偏好数据中学习。相比传统RLHF，DPO更简单、更稳定。

    ### DPO的优势

    - ✅ 无需训练奖励模型
    - ✅ 训练更稳定
    - ✅ 实现更简单
    - ✅ 计算效率更高

    ### 数据格式

    DPO需要偏好对数据：

    ```python
    {
        "prompt": "用户问题",
        "chosen": "更好的回答",
        "rejected": "较差的回答"
    }
    ```
    """
    )
    return


@app.cell
def _():
    print("=" * 60)
    print("🎯 DPO训练示例")
    print("=" * 60)

    # 准备DPO示例数据
    dpo_example_data = [
        {
            "prompt": "什么是人工智能？",
            "chosen": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "rejected": "人工智能就是机器人。"
        },
        {
            "prompt": "如何保持健康？",
            "chosen": "保持健康需要均衡饮食、规律运动、充足睡眠和良好的心理状态。",
            "rejected": "多吃就行了。"
        }
    ]

    print(f"\n✅ 准备了 {len(dpo_example_data)} 条偏好对数据")
    print("\n示例数据:")
    for dpo_idx, dpo_item in enumerate(dpo_example_data, 1):
        print(f"\n{dpo_idx}. 问题: {dpo_item['prompt']}")
        print(f"   ✅ 好回答: {dpo_item['chosen'][:50]}...")
        print(f"   ❌ 差回答: {dpo_item['rejected'][:50]}...")

    # DPO训练配置示例
    dpo_config_example = """
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("your-sft-model")
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

# 2. 配置DPO参数
config = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-7,
    beta=0.1,  # DPO温度参数
    max_length=512,
    max_prompt_length=256,
)

# 3. 创建DPO训练器
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

# 4. 开始训练
trainer.train()
"""

    print("\n" + "=" * 60)
    print("📋 DPO训练配置示例:")
    print("=" * 60)
    print(dpo_config_example)

    return dpo_config_example, dpo_example_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3️⃣ 强化学习 (PPO/RLOO)

    ### 什么是PPO？

    PPO (Proximal Policy Optimization) 是一种强化学习算法，用于从奖励信号中优化模型。这是经典RLHF的核心算法。

    ### PPO vs RLOO

    | 特性 | PPO | RLOO |
    |------|-----|------|
    | 复杂度 | 高 | 低 |
    | 稳定性 | 好 | 很好 |
    | 计算成本 | 高 | 中等 |
    | 适用场景 | 复杂任务 | 简单任务 |

    ### RLHF流程

    ```
    1. SFT训练 → 2. 奖励模型训练 → 3. PPO优化 → 4. 评估
    ```

    ### 奖励函数类型

    - **模型奖励**: 使用训练好的奖励模型
    - **规则奖励**: 基于规则的奖励函数
    - **混合奖励**: 结合多种奖励信号
    """
    )
    return


@app.cell
def _():
    print("=" * 60)
    print("🎮 PPO训练示例")
    print("=" * 60)

    # PPO训练配置示例
    ppo_config_example = """
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# 1. 加载模型（带价值头）
model = AutoModelForCausalLMWithValueHead.from_pretrained("your-sft-model")
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

# 2. 加载奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained("reward-model")

# 3. 配置PPO参数
config = PPOConfig(
    model_name="ppo_model",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    max_grad_norm=0.5,
)

# 4. 创建PPO训练器
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
)

# 5. 训练循环
for batch in dataloader:
    # 生成响应
    response_tensors = ppo_trainer.generate(batch["query"])

    # 计算奖励
    rewards = compute_rewards(response_tensors, reward_model)

    # PPO更新
    stats = ppo_trainer.step(batch["query"], response_tensors, rewards)
"""

    print("\n📋 PPO训练配置示例:")
    print("=" * 60)
    print(ppo_config_example)

    print("\n" + "=" * 60)
    print("💡 PPO训练要点:")
    print("=" * 60)
    print("1. 需要先训练SFT模型")
    print("2. 需要训练奖励模型")
    print("3. 计算成本较高（需要多次前向传播）")
    print("4. 超参数调优很重要")
    print("5. 建议使用分布式训练")

    return (ppo_config_example,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4️⃣ 奖励模型训练

    ### 什么是奖励模型？

    奖励模型是一个分类器，用于评估生成文本的质量。它是RLHF流程中的关键组件。

    ### 奖励模型的作用

    - 评估响应质量
    - 提供训练信号
    - 引导模型优化方向

    ### 训练数据格式

    ```python
    {
        "prompt": "用户问题",
        "chosen": "高质量回答",
        "rejected": "低质量回答"
    }
    ```
    """
    )
    return


@app.cell
def _():
    print("=" * 60)
    print("🏆 奖励模型训练示例")
    print("=" * 60)

    # 奖励模型训练配置
    reward_config_example = """
from trl import RewardConfig, RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载基础模型
model = AutoModelForSequenceClassification.from_pretrained(
    "your-base-model",
    num_labels=1  # 奖励模型输出单个分数
)
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# 2. 配置训练参数
config = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_length=512,
)

# 3. 创建奖励训练器
trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

# 4. 开始训练
trainer.train()

# 5. 使用奖励模型
def get_reward(text):
    inputs = tokenizer(text, return_tensors="pt")
    reward = model(**inputs).logits[0, 0].item()
    return reward
"""

    print("\n📋 奖励模型训练配置:")
    print("=" * 60)
    print(reward_config_example)

    print("\n" + "=" * 60)
    print("💡 奖励模型训练要点:")
    print("=" * 60)
    print("1. 需要高质量的偏好数据")
    print("2. 数据量要足够（建议>10K对）")
    print("3. 注意过拟合问题")
    print("4. 可以使用预训练模型初始化")
    print("5. 评估时关注准确率和校准度")

    return (reward_config_example,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5️⃣ 其他优化方法

    ### KTO (Kahneman-Tversky Optimization)

    基于前景理论的偏好优化方法，不需要成对的偏好数据。

    ```python
    from trl import KTOConfig, KTOTrainer

    config = KTOConfig(
        output_dir="./kto_output",
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
    )
    ```

    ### ORPO (Odds Ratio Preference Optimization)

    结合SFT和偏好优化的单阶段方法。

    ```python
    from trl import ORPOConfig, ORPOTrainer

    config = ORPOConfig(
        output_dir="./orpo_output",
        beta=0.1,
        max_length=512,
    )
    ```

    ### CPO (Contrastive Preference Optimization)

    使用对比学习的偏好优化方法。

    ```python
    from trl import CPOConfig, CPOTrainer

    config = CPOConfig(
        output_dir="./cpo_output",
        beta=0.1,
        label_smoothing=0.0,
    )
    ```

    ### GRPO (Group Relative Policy Optimization)

    组相对策略优化，适用于数学推理等任务。

    ```python
    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir="./grpo_output",
        num_generations=4,
        temperature=0.7,
    )
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 6️⃣ PEFT集成

    ### 什么是PEFT？

    PEFT (Parameter-Efficient Fine-Tuning) 是一种高效微调方法，只训练少量参数。

    ### 支持的PEFT方法

    | 方法 | 说明 | 参数量 |
    |------|------|--------|
    | LoRA | 低秩适应 | ~0.1% |
    | QLoRA | 量化LoRA | ~0.1% |
    | Prefix Tuning | 前缀微调 | ~0.1% |
    | P-Tuning | 提示微调 | ~0.01% |

    ### TRL + PEFT示例

    ```python
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # 1. 配置LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # 2. 创建训练器（自动应用PEFT）
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 7️⃣ 分布式训练

    ### DeepSpeed集成

    TRL完全支持DeepSpeed的ZeRO优化。

    ```bash
    # 使用DeepSpeed启动训练
    accelerate launch --config_file deepspeed_config.yaml train.py
    ```

    **DeepSpeed配置示例 (ds_config.json):**

    ```json
    {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu"
            }
        },
        "fp16": {
            "enabled": true
        }
    }
    ```

    ### FSDP (Fully Sharded Data Parallel)

    ```python
    from trl import SFTConfig

    config = SFTConfig(
        output_dir="./output",
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"
        }
    )
    ```

    ### 多GPU训练

    ```bash
    # 使用accelerate
    accelerate launch --num_processes 4 train.py

    # 使用torchrun
    torchrun --nproc_per_node 4 train.py
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 8️⃣ 内存优化技巧

    ### 1. 梯度检查点 (Gradient Checkpointing)

    ```python
    from trl import SFTConfig

    config = SFTConfig(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ```

    ### 2. 混合精度训练

    ```python
    config = SFTConfig(
        fp16=True,  # 或 bf16=True
        optim="adamw_torch_fused",
    )
    ```

    ### 3. Flash Attention

    ```python
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "model_name",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    ```

    ### 4. 量化训练 (QLoRA)

    ```python
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "model_name",
        quantization_config=bnb_config,
    )
    ```

    ### 内存使用对比

    | 方法 | 7B模型显存 | 13B模型显存 |
    |------|-----------|------------|
    | 全精度 | ~28GB | ~52GB |
    | FP16 | ~14GB | ~26GB |
    | QLoRA | ~6GB | ~10GB |
    | QLoRA + GC | ~4GB | ~7GB |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 9️⃣ 实战案例：完整的RLHF流程

    ### 流程概览

    ```
    原始模型 → SFT → 奖励模型 → PPO/DPO → 对齐模型
    ```

    ### 步骤详解

    #### 第1步：监督微调 (SFT)

    使用指令数据微调基础模型。

    #### 第2步：收集偏好数据

    - 人工标注
    - AI辅助标注
    - 自动生成

    #### 第3步：训练奖励模型

    使用偏好数据训练奖励模型。

    #### 第4步：强化学习优化

    使用PPO或DPO优化模型。

    #### 第5步：评估和迭代

    - 自动评估（困惑度、奖励分数）
    - 人工评估（质量、安全性）
    - 迭代优化
    """
    )
    return


@app.cell
def _():
    print("=" * 60)
    print("🚀 完整RLHF流程示例")
    print("=" * 60)

    rlhf_pipeline = """
# ========== 第1步：SFT训练 ==========
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
)

sft_trainer = SFTTrainer(
    model=base_model,
    args=sft_config,
    train_dataset=instruction_dataset,
)
sft_trainer.train()

# ========== 第2步：训练奖励模型 ==========
from trl import RewardConfig, RewardTrainer

reward_config = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
)

reward_trainer = RewardTrainer(
    model=reward_base_model,
    args=reward_config,
    train_dataset=preference_dataset,
)
reward_trainer.train()

# ========== 第3步：DPO优化（推荐） ==========
from trl import DPOConfig, DPOTrainer

dpo_config = DPOConfig(
    output_dir="./dpo_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    beta=0.1,
)

dpo_trainer = DPOTrainer(
    model=sft_model,
    args=dpo_config,
    train_dataset=preference_dataset,
)
dpo_trainer.train()

# ========== 或使用PPO优化 ==========
from trl import PPOConfig, PPOTrainer

ppo_config = PPOConfig(
    model_name="ppo_model",
    learning_rate=1.41e-5,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=sft_model_with_value_head,
)

# PPO训练循环
for batch in dataloader:
    responses = ppo_trainer.generate(batch["query"])
    rewards = reward_model(responses)
    ppo_trainer.step(batch["query"], responses, rewards)
"""

    print("\n📋 完整RLHF流程代码:")
    print("=" * 60)
    print(rlhf_pipeline)

    print("\n" + "=" * 60)
    print("💡 流程要点:")
    print("=" * 60)
    print("1. SFT是基础，质量决定上限")
    print("2. 偏好数据质量很关键")
    print("3. DPO比PPO更简单稳定（推荐）")
    print("4. 需要充分的评估和测试")
    print("5. 可以多次迭代优化")

    return (rlhf_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔟 CLI工具使用

    ### TRL CLI

    TRL提供了命令行工具，简化训练流程。

    #### SFT训练

    ```bash
    trl sft \\
        --model_name_or_path Qwen/Qwen2.5-0.5B \\
        --dataset_name timdettmers/openassistant-guanaco \\
        --output_dir ./sft_output \\
        --num_train_epochs 3 \\
        --per_device_train_batch_size 4 \\
        --learning_rate 2e-5
    ```

    #### DPO训练

    ```bash
    trl dpo \\
        --model_name_or_path ./sft_model \\
        --dataset_name Anthropic/hh-rlhf \\
        --output_dir ./dpo_output \\
        --num_train_epochs 1 \\
        --beta 0.1
    ```

    #### 聊天界面

    ```bash
    trl chat --model_name_or_path ./trained_model
    ```

    ### 配置文件方式

    ```yaml
    # config.yaml
    model_name_or_path: Qwen/Qwen2.5-0.5B
    dataset_name: timdettmers/openassistant-guanaco
    output_dir: ./output
    num_train_epochs: 3
    per_device_train_batch_size: 4
    learning_rate: 2.0e-5
    gradient_checkpointing: true
    fp16: true
    ```

    ```bash
    trl sft --config config.yaml
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 API速查表

    ### 训练器 (Trainers)

    | 训练器 | 用途 | 关键参数 |
    |--------|------|----------|
    | `SFTTrainer` | 监督微调 | `max_seq_length`, `packing` |
    | `DPOTrainer` | 直接偏好优化 | `beta`, `max_length` |
    | `PPOTrainer` | 强化学习 | `ppo_epochs`, `mini_batch_size` |
    | `RewardTrainer` | 奖励模型 | `max_length` |
    | `KTOTrainer` | KT优化 | `desirable_weight` |
    | `ORPOTrainer` | OR优化 | `beta` |
    | `GRPOTrainer` | 组相对优化 | `num_generations` |

    ### 配置类 (Configs)

    | 配置 | 说明 | 示例 |
    |------|------|------|
    | `SFTConfig` | SFT配置 | `SFTConfig(output_dir="./output")` |
    | `DPOConfig` | DPO配置 | `DPOConfig(beta=0.1)` |
    | `PPOConfig` | PPO配置 | `PPOConfig(learning_rate=1e-5)` |
    | `RewardConfig` | 奖励模型配置 | `RewardConfig(num_train_epochs=1)` |

    ### 模型类

    | 类 | 说明 |
    |-----|------|
    | `AutoModelForCausalLMWithValueHead` | 带价值头的因果语言模型 |
    | `AutoModelForSeq2SeqLMWithValueHead` | 带价值头的序列到序列模型 |

    ### 工具函数

    | 函数 | 说明 |
    |------|------|
    | `create_reference_model()` | 创建参考模型 |
    | `prepare_model_for_kbit_training()` | 准备量化训练 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 💡 最佳实践

    ### 1. 选择合适的训练方法

    - **简单任务** → SFT即可
    - **需要对齐** → SFT + DPO
    - **复杂对齐** → SFT + 奖励模型 + PPO
    - **资源受限** → 使用PEFT (LoRA/QLoRA)

    ### 2. 数据准备

    - 确保数据质量高于数量
    - SFT数据：多样性很重要
    - 偏好数据：差异要明显
    - 数据清洗和去重

    ### 3. 超参数调优

    **SFT关键参数:**
    - `learning_rate`: 2e-5 到 5e-5
    - `num_train_epochs`: 1-3
    - `max_seq_length`: 根据任务调整

    **DPO关键参数:**
    - `beta`: 0.1 到 0.5
    - `learning_rate`: 5e-7 到 5e-6
    - `num_train_epochs`: 1

    **PPO关键参数:**
    - `learning_rate`: 1e-5 到 5e-5
    - `ppo_epochs`: 4
    - `mini_batch_size`: 根据显存调整

    ### 4. 评估策略

    - 使用验证集监控过拟合
    - 定期生成样本检查质量
    - 使用自动评估指标（BLEU、ROUGE等）
    - 人工评估关键样本

    ### 5. 常见问题

    **显存不足:**
    - 使用梯度检查点
    - 减小batch size
    - 使用QLoRA
    - 使用DeepSpeed ZeRO

    **训练不稳定:**
    - 降低学习率
    - 使用梯度裁剪
    - 检查数据质量
    - 使用warmup

    **效果不好:**
    - 增加训练数据
    - 调整超参数
    - 检查数据分布
    - 尝试不同的基础模型
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔗 集成示例

    ### 与Hugging Face Hub集成

    ```python
    from trl import SFTTrainer, SFTConfig

    config = SFTConfig(
        output_dir="./output",
        push_to_hub=True,
        hub_model_id="username/model-name",
        hub_strategy="every_save",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.push_to_hub()
    ```

    ### 与Weights & Biases集成

    ```python
    import wandb

    wandb.init(project="trl-training", name="sft-run-1")

    config = SFTConfig(
        output_dir="./output",
        report_to="wandb",
        logging_steps=10,
    )
    ```

    ### 与vLLM集成（推理加速）

    ```python
    from trl import SFTTrainer
    from trl.trainer.utils import get_vllm_model

    # 训练后使用vLLM加速推理
    vllm_model = get_vllm_model(
        model_name="./trained_model",
        tensor_parallel_size=2,
    )

    outputs = vllm_model.generate(
        prompts=["你好"],
        max_tokens=100,
    )
    ```

    ### 与Unsloth集成（训练加速）

    ```python
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # 使用TRL训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📊 性能对比

    ### 训练方法对比

    | 方法 | 训练时间 | 显存占用 | 效果 | 难度 |
    |------|---------|---------|------|------|
    | SFT | 基准 | 基准 | ⭐⭐⭐ | ⭐ |
    | SFT + LoRA | 0.8x | 0.3x | ⭐⭐⭐ | ⭐ |
    | SFT + QLoRA | 0.9x | 0.2x | ⭐⭐⭐ | ⭐⭐ |
    | DPO | 1.2x | 1.5x | ⭐⭐⭐⭐ | ⭐⭐ |
    | PPO | 3-5x | 2-3x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

    ### 模型规模与资源需求

    | 模型大小 | 全精度SFT | QLoRA | 推荐GPU |
    |---------|----------|-------|---------|
    | 1B | 4GB | 2GB | RTX 3060 |
    | 3B | 12GB | 4GB | RTX 3090 |
    | 7B | 28GB | 6GB | A100 40GB |
    | 13B | 52GB | 10GB | A100 80GB |
    | 70B | 280GB | 40GB | 8x A100 |

    ### 优化技术效果

    | 技术 | 显存节省 | 速度影响 | 精度影响 |
    |------|---------|---------|---------|
    | Gradient Checkpointing | 30-50% | -20% | 无 |
    | Flash Attention 2 | 10-20% | +30% | 无 |
    | 4-bit量化 | 75% | -10% | 轻微 |
    | DeepSpeed ZeRO-3 | 60-80% | -5% | 无 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🎓 总结

    ### TRL的优势

    - ✅ **全栈解决方案** - 从SFT到RLHF的完整工具链
    - ✅ **易于使用** - 简洁的API和丰富的文档
    - ✅ **灵活性高** - 支持多种训练方法和优化技术
    - ✅ **生态完善** - 与Transformers、PEFT、DeepSpeed等无缝集成
    - ✅ **持续更新** - Hugging Face官方维护，紧跟最新研究

    ### 适用场景

    - 🎯 **指令微调** - 训练遵循指令的模型
    - 💬 **对话模型** - 构建聊天机器人
    - 🔍 **偏好对齐** - 使模型输出符合人类偏好
    - 📚 **领域适应** - 特定领域的模型微调
    - 🧪 **研究实验** - 快速验证新想法

    ### 学习路径

    1. **入门阶段**
       - 学习SFT基础
       - 理解数据格式
       - 运行简单示例

    2. **进阶阶段**
       - 掌握DPO训练
       - 学习PEFT集成
       - 优化训练性能

    3. **高级阶段**
       - 实现完整RLHF
       - 自定义训练流程
       - 分布式训练部署

    ### 何时使用TRL

    - ✅ **需要微调语言模型** → 使用TRL
    - ✅ **需要偏好对齐** → 使用TRL的DPO/PPO
    - ✅ **资源受限** → 使用TRL + PEFT
    - ✅ **快速原型** → 使用TRL CLI

    ### 何时不使用TRL

    - ❌ **只需要推理** → 使用vLLM或TGI
    - ❌ **预训练大模型** → 使用Megatron-LM
    - ❌ **非语言模型** → 使用其他框架

    ### 学习资源

    - 📖 [官方文档](https://huggingface.co/docs/trl)
    - 💻 [GitHub仓库](https://github.com/huggingface/trl)
    - 📝 [示例代码](https://github.com/huggingface/trl/tree/main/examples)
    - 🎥 [视频教程](https://huggingface.co/learn)
    - 💬 [Discord社区](https://discord.gg/hugging-face)
    - 📚 [Smol Course](https://huggingface.co/learn/smol-course)

    ### 相关工具

    - **Transformers** - 基础模型库
    - **PEFT** - 参数高效微调
    - **Accelerate** - 分布式训练
    - **DeepSpeed** - 大规模训练优化
    - **vLLM** - 高性能推理
    - **Unsloth** - 训练加速

    ---

    **恭喜！** 🎉 你已经掌握了TRL的核心概念和使用方法。

    现在你可以开始训练自己的语言模型了！

    ### 下一步建议

    1. 选择一个小模型（如Qwen2.5-0.5B）进行SFT实验
    2. 准备高质量的训练数据
    3. 使用LoRA减少资源需求
    4. 逐步尝试DPO等高级方法
    5. 加入社区，分享经验和问题
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📖 附录：常用代码片段

    ### 快速开始模板

    ```python
    # 最小化SFT训练示例
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # 加载模型和数据
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

    # 配置和训练
    config = SFTConfig(output_dir="./output", num_train_epochs=1)
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    ```

    ### 数据处理模板

    ```python
    # 格式化对话数据
    def format_chat_template(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    dataset = dataset.map(format_chat_template)
    ```

    ### 推理模板

    ```python
    # 使用训练好的模型
    from transformers import pipeline

    generator = pipeline("text-generation", model="./output")
    output = generator("你好", max_length=100)
    print(output[0]["generated_text"])
    ```

    ### 评估模板

    ```python
    # 计算困惑度
    from trl import SFTTrainer

    eval_results = trainer.evaluate()
    print(f"Perplexity: {eval_results['eval_loss']:.2f}")
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()


