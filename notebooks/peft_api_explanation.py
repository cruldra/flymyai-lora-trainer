import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(
        r"""
        # PEFT 库 API 深度解析 - 参数高效微调

        这个笔记本将深入解释 `train.py` 中使用的 PEFT（Parameter Efficient Fine-Tuning）库的各种 API，
        重点讲解 LoRA（Low-Rank Adaptation）技术的原理和实现。

        ## 什么是 PEFT？

        PEFT 是一种参数高效的微调技术，它允许你在不修改原始模型大部分参数的情况下，
        通过添加少量可训练参数来适应新任务。

        **核心优势**：
        
        - **显存效率**：只训练少量参数，大幅减少显存需求
        - **存储效率**：只需保存适配器权重，而不是整个模型
        - **训练速度**：参数少，训练更快
        - **避免灾难性遗忘**：保持原模型能力的同时学习新任务
        """
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. LoRA 技术原理

        ### 什么是 LoRA？

        LoRA（Low-Rank Adaptation）是一种将大矩阵分解为两个小矩阵的技术。

        **传统微调的问题**：
        ```python
        # 假设有一个线性层
        linear = nn.Linear(4096, 4096)  # 参数量：4096 * 4096 = 16M
        
        # 传统微调：更新所有16M参数
        for param in linear.parameters():
            param.requires_grad = True  # 所有参数都要训练
        ```

        **LoRA 的解决方案**：
        ```python
        # 原始权重矩阵 W (4096 x 4096) 保持冻结
        # 添加两个小矩阵：
        # A: (4096 x r)  其中 r << 4096，比如 r=64
        # B: (r x 4096)
        # 更新后的权重 = W + B @ A
        
        # 参数量对比：
        # 原始：4096 * 4096 = 16M
        # LoRA：4096 * 64 + 64 * 4096 = 0.5M  (减少97%！)
        ```

        **为什么这样有效？**
        
        - 大多数深度学习任务的适应只需要低秩的权重更新
        - 通过限制秩（rank），强制模型学习最重要的特征
        - 保持原模型的泛化能力
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. LoraConfig - 配置 LoRA 参数

        ```python
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        ```

        ### 参数详解

        #### r (rank) - 秩参数
        **作用**：控制 LoRA 矩阵的维度大小
        
        ```python
        # 如果原始矩阵是 (4096, 4096)，r=64
        # LoRA 分解为：
        # A: (4096, 64)  - 下投影矩阵
        # B: (64, 4096)  - 上投影矩阵
        # 新权重 = 原权重 + B @ A
        ```

        **选择原则**：
        
        - **r 越小**：参数越少，训练越快，但表达能力有限
        - **r 越大**：表达能力强，但参数增多，接近全量微调
        - **常用值**：8, 16, 32, 64, 128

        #### lora_alpha - 缩放因子
        **作用**：控制 LoRA 更新的强度
        
        ```python
        # 实际更新 = (lora_alpha / r) * (B @ A)
        # 如果 lora_alpha = r，则缩放因子 = 1
        # 如果 lora_alpha = 2 * r，则 LoRA 更新会被放大2倍
        ```

        **为什么需要缩放？**
        
        - 平衡原始权重和 LoRA 更新的贡献
        - 防止 LoRA 更新过小而被忽略
        - 通常设置为与 r 相等或稍大
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### init_lora_weights - 初始化策略
        **作用**：决定 LoRA 矩阵的初始化方式
        
        ```python
        init_lora_weights="gaussian"  # 高斯分布初始化
        ```

        **可选策略**：
        
        - **"gaussian"**：A 矩阵用高斯分布，B 矩阵用零初始化
        - **"xavier"**：Xavier 均匀分布初始化
        - **"kaiming"**：Kaiming 初始化
        - **False**：不进行特殊初始化

        **为什么 B 矩阵初始化为零？**
        ```python
        # 训练开始时：
        # A: 随机初始化 (gaussian)
        # B: 零初始化
        # 因此：B @ A = 0
        # 新权重 = 原权重 + 0 = 原权重
        # 这确保训练开始时模型行为与原模型完全一致
        ```

        #### target_modules - 目标模块
        **作用**：指定哪些模块应用 LoRA
        
        ```python
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        ```

        **这些模块是什么？**
        
        - **to_q**：Query 投影层（注意力机制中的查询）
        - **to_k**：Key 投影层（注意力机制中的键）
        - **to_v**：Value 投影层（注意力机制中的值）
        - **to_out.0**：输出投影层

        **为什么选择这些模块？**
        
        - 注意力层是 Transformer 的核心，影响最大
        - 这些线性层参数量大，LoRA 效果明显
        - 实验证明在这些层应用 LoRA 效果最好
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. add_adapter() - 添加 LoRA 适配器

        ```python
        flux_transformer.add_adapter(lora_config)
        ```

        **这一行代码做了什么？**

        ### 模型结构变化
        ```python
        # 原始模型结构：
        # Linear(in_features=4096, out_features=4096)
        
        # 添加 LoRA 后：
        # Linear(in_features=4096, out_features=4096) + LoRA适配器
        #   ├── 原始权重 W (冻结)
        #   ├── LoRA_A (4096 x 64, 可训练)
        #   └── LoRA_B (64 x 4096, 可训练)
        ```

        ### 前向传播变化
        ```python
        # 原始前向传播：
        # output = input @ W
        
        # 添加 LoRA 后：
        # output = input @ (W + lora_alpha/r * (LoRA_B @ LoRA_A))
        #        = input @ W + input @ (lora_alpha/r * LoRA_B @ LoRA_A)
        #        = 原始输出 + LoRA贡献
        ```

        **为什么这样设计？**
        
        - **保持兼容性**：原始模型结构不变
        - **模块化**：LoRA 可以独立保存和加载
        - **可组合性**：可以同时使用多个适配器
        - **零开销推理**：可以将 LoRA 权重合并到原权重中
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 4. 参数冻结和训练设置

        ```python
        for n, param in flux_transformer.named_parameters():
            if 'lora' not in n:
                param.requires_grad = False  # 冻结非LoRA参数
            else:
                param.requires_grad = True   # 只训练LoRA参数
                print(n)
        ```

        **为什么要这样设置？**

        ### 参数分类
        ```python
        # 模型中的参数分为两类：
        
        # 1. 原始参数（冻结）
        # transformer.layers.0.attention.to_q.weight
        # transformer.layers.0.attention.to_k.weight
        # transformer.layers.0.attention.to_v.weight
        
        # 2. LoRA参数（可训练）
        # transformer.layers.0.attention.to_q.lora_A.weight
        # transformer.layers.0.attention.to_q.lora_B.weight
        # transformer.layers.0.attention.to_k.lora_A.weight
        # transformer.layers.0.attention.to_k.lora_B.weight
        ```

        ### 参数量对比
        ```python
        # 假设模型有1B参数
        # 传统微调：1B参数全部训练
        # LoRA微调：只训练约1-10M参数（减少99%）
        
        print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
        # 输出类似：5.2 parameters (即520万可训练参数)
        ```

        **优势分析**：
        
        - **显存节省**：只需为少量参数计算梯度
        - **训练加速**：反向传播计算量大幅减少
        - **避免过拟合**：参数少，不容易过拟合小数据集
        - **保持泛化**：原始能力得到保留
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 5. 优化器设置

        ```python
        lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
        optimizer = optimizer_cls(
            lora_layers,  # 只优化LoRA参数
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        ```

        **为什么只传入 LoRA 参数？**

        ### 优化器工作原理
        ```python
        # 优化器只会更新传入的参数
        # 如果传入所有参数：
        all_params = flux_transformer.parameters()
        optimizer = AdamW(all_params)  # 会尝试更新所有参数
        
        # 即使设置了 requires_grad=False，优化器仍会为这些参数分配内存
        # 浪费显存和计算资源
        ```

        **正确的做法**：
        ```python
        # 只传入需要训练的参数
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(trainable_params)
        
        # 优势：
        # 1. 节省显存：不为冻结参数分配优化器状态
        # 2. 提高效率：减少优化器的计算开销
        # 3. 避免错误：确保只更新想要训练的参数
        ```

        **AdamW 优化器状态**：
        ```python
        # AdamW 为每个参数维护两个状态：
        # - 一阶矩估计 (momentum)
        # - 二阶矩估计 (variance)
        # 如果有1B参数，优化器状态需要额外2B参数的显存
        # 通过只传入LoRA参数，显存需求从2B降到几MB
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 6. 模型保存 - get_peft_model_state_dict()

        ```python
        unwrapped_flux_transformer = unwrap_model(flux_transformer)
        flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_flux_transformer)
        )
        ```

        **为什么要这样保存？**

        ### 传统模型保存的问题
        ```python
        # 传统做法：保存整个模型
        torch.save(model.state_dict(), "model.pt")
        # 问题：文件很大（几GB到几十GB）
        # 对于1B参数模型，需要4GB存储空间
        ```

        ### PEFT 的优雅解决方案
        ```python
        # 只保存 LoRA 参数
        lora_state_dict = get_peft_model_state_dict(model)
        torch.save(lora_state_dict, "lora_adapter.pt")
        # 优势：文件很小（几MB到几十MB）
        # 对于相同模型，LoRA适配器只需要几MB
        ```

        **get_peft_model_state_dict() 做了什么？**
        ```python
        # 从完整的 state_dict 中提取 LoRA 参数
        full_state_dict = {
            'transformer.layers.0.attention.to_q.weight': tensor(...),           # 原始权重
            'transformer.layers.0.attention.to_q.lora_A.weight': tensor(...),   # LoRA A
            'transformer.layers.0.attention.to_q.lora_B.weight': tensor(...),   # LoRA B
            # ... 更多参数
        }

        # 提取后只保留 LoRA 参数
        lora_state_dict = {
            'transformer.layers.0.attention.to_q.lora_A.weight': tensor(...),
            'transformer.layers.0.attention.to_q.lora_B.weight': tensor(...),
            # 只有 LoRA 相关参数
        }
        ```

        **存储效率对比**：

        - **完整模型**：1B参数 × 4字节 = 4GB
        - **LoRA适配器**：5M参数 × 4字节 = 20MB
        - **压缩比**：200:1
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 7. convert_state_dict_to_diffusers() - 格式转换

        ```python
        flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_flux_transformer)
        )
        ```

        **为什么需要格式转换？**

        ### PEFT 格式 vs Diffusers 格式
        ```python
        # PEFT 原始格式：
        peft_format = {
            'base_model.model.transformer.layers.0.attention.to_q.lora_A.weight': tensor(...),
            'base_model.model.transformer.layers.0.attention.to_q.lora_B.weight': tensor(...),
        }

        # Diffusers 期望格式：
        diffusers_format = {
            'transformer.layers.0.attention.to_q.lora_A.weight': tensor(...),
            'transformer.layers.0.attention.to_q.lora_B.weight': tensor(...),
        }
        ```

        **转换的必要性**：

        - **兼容性**：确保保存的权重可以被 Diffusers 库正确加载
        - **标准化**：统一不同库之间的命名约定
        - **互操作性**：允许在不同框架间共享模型

        ### 实际使用场景
        ```python
        # 训练完成后，你可以这样使用：

        # 1. 加载基础模型
        pipeline = QwenImagePipeline.from_pretrained("base_model")

        # 2. 加载 LoRA 权重
        pipeline.load_lora_weights("path/to/lora_adapter")

        # 3. 直接使用
        image = pipeline("a beautiful landscape")
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 8. LoRA 的实际工作流程

        ### 训练阶段
        ```python
        # 1. 配置 LoRA
        lora_config = LoraConfig(r=64, lora_alpha=64, target_modules=["to_q", "to_k", "to_v"])

        # 2. 添加适配器
        model.add_adapter(lora_config)

        # 3. 冻结原始参数
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

        # 4. 训练（只更新 LoRA 参数）
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()))

        # 5. 保存适配器
        lora_state_dict = get_peft_model_state_dict(model)
        torch.save(lora_state_dict, "adapter.pt")
        ```

        ### 推理阶段
        ```python
        # 1. 加载基础模型
        model = BaseModel.from_pretrained("base_model")

        # 2. 加载 LoRA 适配器
        model.load_adapter("adapter.pt")

        # 3. 推理
        output = model(input_data)
        ```

        ### 权重合并（可选）
        ```python
        # 为了推理效率，可以将 LoRA 权重合并到原始权重中
        model.merge_adapter()

        # 合并后：
        # 新权重 = 原始权重 + LoRA权重
        # 推理时不需要额外计算，速度更快
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 9. LoRA 的优势和局限性

        ### 优势总结

        #### 1. 参数效率
        ```python
        # 参数量对比（以1B参数模型为例）
        full_finetune = 1_000_000_000  # 1B参数
        lora_params = 5_000_000        # 5M参数
        reduction = full_finetune / lora_params  # 200倍减少
        ```

        #### 2. 存储效率
        ```python
        # 存储空间对比
        full_model_size = "4GB"      # 完整模型
        lora_adapter_size = "20MB"   # LoRA适配器
        # 可以为不同任务保存多个适配器，而不是多个完整模型
        ```

        #### 3. 训练效率
        ```python
        # 显存使用对比
        full_finetune_memory = "24GB"  # 全量微调
        lora_memory = "8GB"            # LoRA微调
        # 可以在更小的GPU上训练大模型
        ```

        #### 4. 避免灾难性遗忘
        ```python
        # 原始模型能力保持不变
        # 只在特定层添加适应性，不破坏预训练知识
        ```

        ### 局限性

        #### 1. 表达能力限制
        ```python
        # 低秩约束可能限制模型适应复杂任务的能力
        # 对于需要大幅改变模型行为的任务，效果可能不如全量微调
        ```

        #### 2. 超参数敏感
        ```python
        # rank (r) 和 lora_alpha 的选择对结果影响很大
        # 需要针对具体任务进行调优
        ```

        #### 3. 推理开销
        ```python
        # 如果不合并权重，推理时需要额外计算
        # output = input @ W + input @ (B @ A)
        # 比原始推理多了一次矩阵乘法
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 10. 实践建议

        ### 选择合适的 rank
        ```python
        # 任务复杂度 vs rank 选择
        simple_tasks = {"rank": 8}      # 简单分类、情感分析
        medium_tasks = {"rank": 32}     # 文本生成、对话
        complex_tasks = {"rank": 128}   # 复杂推理、多模态任务
        ```

        ### 选择目标模块
        ```python
        # 不同模型架构的推荐设置

        # Transformer (BERT/GPT类)
        transformer_targets = ["query", "key", "value", "dense"]

        # Vision Transformer
        vit_targets = ["qkv", "proj"]

        # Diffusion Models (如本例)
        diffusion_targets = ["to_q", "to_k", "to_v", "to_out.0"]
        ```

        ### 学习率设置
        ```python
        # LoRA 通常需要比全量微调更高的学习率
        full_finetune_lr = 1e-5
        lora_lr = 1e-4  # 通常是全量微调的5-10倍
        ```

        ### 监控训练
        ```python
        # 关键指标
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        efficiency = trainable_params / total_params

        print(f"训练参数: {trainable_params:,}")
        print(f"总参数: {total_params:,}")
        print(f"参数效率: {efficiency:.2%}")
        ```

        ### 多任务适配
        ```python
        # 可以为不同任务训练不同的适配器
        model.load_adapter("task1_adapter.pt", adapter_name="task1")
        model.load_adapter("task2_adapter.pt", adapter_name="task2")

        # 切换任务
        model.set_adapter("task1")  # 使用任务1的适配器
        output1 = model(input_data)

        model.set_adapter("task2")  # 切换到任务2
        output2 = model(input_data)
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 11. 总结：PEFT 的核心价值

        ### 解决的核心问题

        #### 1. 资源限制
        **问题**：大模型微调需要大量GPU显存和存储空间

        **解决方案**：通过参数高效技术，用1%的参数达到90%的效果

        #### 2. 部署效率
        **问题**：为每个任务保存完整模型，存储和传输成本高

        **解决方案**：一个基础模型 + 多个轻量适配器

        #### 3. 知识保持
        **问题**：全量微调可能导致灾难性遗忘

        **解决方案**：保持原始权重不变，只添加任务特定的适应层

        ### PEFT vs 传统微调

        | 方面 | 传统微调 | PEFT (LoRA) |
        |------|----------|-------------|
        | 参数量 | 100% | 1-5% |
        | 显存需求 | 高 | 低 |
        | 训练速度 | 慢 | 快 |
        | 存储空间 | 大 | 小 |
        | 灾难性遗忘 | 可能发生 | 很少发生 |
        | 多任务部署 | 困难 | 容易 |

        ### 适用场景

        **推荐使用 PEFT 的情况**：

        - 资源受限的环境
        - 需要快速适配多个任务
        - 数据集相对较小
        - 需要保持原模型能力

        **考虑全量微调的情况**：

        - 有充足的计算资源
        - 任务与预训练差异很大
        - 数据集非常大
        - 对性能要求极高

        PEFT 代表了深度学习微调的未来方向：**用更少的资源，实现更好的效果**。
        """
    )
    return


if __name__ == "__main__":
    app.run()
