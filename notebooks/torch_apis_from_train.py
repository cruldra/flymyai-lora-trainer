import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(
        r"""
        # PyTorch API 深度解析 - 深度学习训练的核心操作

        这个笔记本将深入解释 `train.py` 中使用的 PyTorch API，
        重点讲解每个操作的目的、原理和在深度学习训练中的作用。

        ## PyTorch 在深度学习中的地位

        PyTorch 是现代深度学习的基础框架，提供了：
        
        - **张量操作**：高效的多维数组计算
        - **自动微分**：自动计算梯度，支持反向传播
        - **GPU 加速**：无缝的 CPU/GPU 计算切换
        - **动态图**：灵活的计算图构建
        - **丰富的 API**：从基础运算到高级优化器

        ## 训练流程中的 PyTorch 操作

        在扩散模型训练中，PyTorch 承担着：
        
        1. **数据处理**：张量创建、形状变换、设备管理
        2. **数值计算**：数学运算、统计操作、损失计算
        3. **梯度管理**：自动微分、梯度控制、参数更新
        4. **性能优化**：混合精度、内存管理、并行计算
        """
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. 数据类型管理 - 为什么精度如此重要？

        ```python
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        ```

        ### 精度选择的影响

        #### torch.float32 - 标准精度
        **特点**：32位浮点数，8位指数，23位尾数
        
        ```python
        # 数值范围：约 ±3.4 × 10^38
        # 精度：约7位有效数字
        # 内存：每个数占4字节
        
        # 优势：
        # - 数值稳定性好
        # - 精度高，适合复杂计算
        # - 兼容性最好
        
        # 劣势：
        # - 内存占用大
        # - 计算速度相对较慢
        ```

        #### torch.float16 - 半精度
        **特点**：16位浮点数，5位指数，10位尾数
        
        ```python
        # 数值范围：约 ±6.5 × 10^4
        # 精度：约3位有效数字
        # 内存：每个数占2字节
        
        # 优势：
        # - 内存减半，可训练更大模型
        # - 现代GPU对fp16有硬件加速
        # - 训练速度提升2倍
        
        # 劣势：
        # - 数值范围小，容易溢出
        # - 精度低，可能影响训练稳定性
        # - 需要特殊处理防止梯度消失
        ```

        #### torch.bfloat16 - 脑浮点
        **特点**：16位浮点数，8位指数，7位尾数
        
        ```python
        # 数值范围：与float32相同
        # 精度：约2位有效数字
        # 内存：每个数占2字节
        
        # 优势：
        # - 数值范围大，不容易溢出
        # - 与float32转换简单（直接截断）
        # - TPU和新GPU有硬件支持
        
        # 劣势：
        # - 精度比fp16更低
        # - 硬件支持不如fp16广泛
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. 张量创建和设备管理 - 为什么要精确控制？

        ```python
        noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
        ```

        ### torch.randn_like() 的设计哲学

        #### 为什么不用 torch.randn()？
        ```python
        # 错误的做法
        noise = torch.randn(2, 4, 8, 16, 16)  # 硬编码形状
        # 问题：
        # 1. 形状可能不匹配
        # 2. 设备可能不一致
        # 3. 数据类型可能不同
        
        # 正确的做法
        noise = torch.randn_like(pixel_latents)  # 自动匹配所有属性
        # 优势：
        # 1. 形状自动匹配
        # 2. 设备自动匹配
        # 3. 数据类型自动匹配
        ```

        #### 显式指定 device 和 dtype 的原因
        ```python
        # 为什么要显式指定？
        noise = torch.randn_like(
            pixel_latents, 
            device=accelerator.device,  # 确保在正确的GPU上
            dtype=weight_dtype          # 确保使用混合精度
        )
        
        # 防止的问题：
        # 1. 设备不匹配：tensor在CPU，模型在GPU
        # 2. 类型不匹配：tensor是float32，模型是float16
        # 3. 性能问题：避免运行时的类型转换
        ```

        ### 设备管理的重要性

        #### 分布式训练中的设备分配
        ```python
        # 在4GPU训练中：
        # 进程0：accelerator.device = cuda:0
        # 进程1：accelerator.device = cuda:1
        # 进程2：accelerator.device = cuda:2
        # 进程3：accelerator.device = cuda:3
        
        # 如果不指定device：
        noise = torch.randn_like(pixel_latents)  # 可能在错误的GPU上
        # 结果：数据传输开销，甚至运行时错误
        ```

        #### 内存效率考虑
        ```python
        # 直接在目标设备创建张量
        noise = torch.randn_like(pixel_latents, device=target_device)
        
        # 而不是先创建再转移
        noise = torch.randn_like(pixel_latents).to(target_device)  # 浪费内存
        # 第二种方法会临时占用两倍内存
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. 张量形状操作 - 为什么要这样变换？

        ### unsqueeze() - 添加维度的艺术

        ```python
        pixel_values = pixel_values.unsqueeze(2)  # 在第2维插入新维度
        ```

        #### 为什么要添加时间维度？
        ```python
        # 原始图像形状：(batch, channels, height, width)
        # 例如：(4, 3, 512, 512)
        
        # 视频/序列处理需要时间维度：(batch, channels, time, height, width)
        # 添加后：(4, 3, 1, 512, 512)
        
        # 原因：
        # 1. 统一接口：图像和视频使用相同的处理管道
        # 2. 模型期望：某些模型架构要求5D输入
        # 3. 扩展性：为未来的时序建模预留接口
        ```

        #### unsqueeze 的位置选择
        ```python
        # 不同位置的含义：
        tensor.unsqueeze(0)   # 在最前面添加batch维度
        tensor.unsqueeze(1)   # 在第1维添加
        tensor.unsqueeze(2)   # 在第2维添加（本例中是时间维度）
        tensor.unsqueeze(-1)  # 在最后面添加
        
        # 选择原则：根据数据的语义含义选择合适位置
        ```

        ### permute() - 维度重排的必要性

        ```python
        pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)  # 交换维度1和2
        ```

        #### 为什么要交换维度？
        ```python
        # VAE编码后：(batch, channels, time, height, width)
        # 例如：(2, 16, 1, 64, 64)
        
        # 某些处理需要：(batch, time, channels, height, width)
        # 交换后：(2, 1, 16, 64, 64)
        
        # 原因：
        # 1. 模型期望：不同模型对维度顺序有不同要求
        # 2. 计算效率：某些操作在特定维度顺序下更高效
        # 3. 内存布局：优化内存访问模式
        ```

        #### 维度顺序的约定
        ```python
        # 图像处理常见约定：
        # PyTorch: (N, C, H, W) - batch, channels, height, width
        # TensorFlow: (N, H, W, C) - batch, height, width, channels
        
        # 视频处理常见约定：
        # (N, C, T, H, W) - batch, channels, time, height, width
        # (N, T, C, H, W) - batch, time, channels, height, width
        
        # 选择原则：与模型架构和库约定保持一致
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### view() 和 reshape() - 张量重塑的细节

        ```python
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 1, vae.config.z_dim, 1, 1)
        ```

        #### view() vs reshape() 的区别
        ```python
        # view() - 要求内存连续
        tensor_view = tensor.view(new_shape)
        # 优势：零拷贝，速度快
        # 限制：只能用于内存连续的张量

        # reshape() - 更灵活
        tensor_reshape = tensor.reshape(new_shape)
        # 优势：总是能工作，必要时会复制内存
        # 劣势：可能涉及内存拷贝
        ```

        #### 为什么要重塑为 (1, 1, z_dim, 1, 1)？
        ```python
        # 原始配置：latents_mean = [0.1, 0.2, 0.3, 0.4]  # 形状: (4,)
        # 目标张量：pixel_latents  # 形状: (batch, time, channels, height, width)
        #                          # 例如: (2, 1, 4, 64, 64)

        # 重塑后：latents_mean  # 形状: (1, 1, 4, 1, 1)

        # 广播机制：
        # (2, 1, 4, 64, 64) - (1, 1, 4, 1, 1) = (2, 1, 4, 64, 64)
        # 每个通道减去对应的均值，空间维度自动广播
        ```

        #### 广播的优势
        ```python
        # 不使用广播的做法：
        mean_expanded = latents_mean.expand(2, 1, 4, 64, 64)  # 显式扩展
        result = pixel_latents - mean_expanded

        # 使用广播的做法：
        mean_reshaped = latents_mean.view(1, 1, 4, 1, 1)  # 只重塑形状
        result = pixel_latents - mean_reshaped  # 自动广播

        # 广播的优势：
        # 1. 内存效率：不需要实际扩展张量
        # 2. 计算效率：GPU可以优化广播操作
        # 3. 代码简洁：自动处理维度匹配
        ```

        ### 动态维度扩展

        ```python
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        ```

        #### 为什么要动态扩展？
        ```python
        # 问题场景：sigma的维度可能不够
        # sigma: (batch_size,) = (2,)
        # 目标: 与 pixel_latents 匹配，形状 (2, 1, 4, 64, 64)

        # 动态扩展过程：
        # 初始: (2,)
        # 第1次: (2, 1)
        # 第2次: (2, 1, 1)
        # 第3次: (2, 1, 1, 1)
        # 第4次: (2, 1, 1, 1, 1)

        # 最终可以与 (2, 1, 4, 64, 64) 进行广播运算
        ```

        #### 通用性设计
        ```python
        # 这种设计的优势：
        # 1. 适应性：可以处理任意维度的张量
        # 2. 鲁棒性：不依赖硬编码的维度数
        # 3. 可扩展性：支持未来的模型架构变化
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 4. 数学运算和标准化 - 数值稳定性的保证

        ### 线性组合运算

        ```python
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        ```

        #### Flow Matching 的数学原理
        ```python
        # 这是 Flow Matching 的核心公式
        # x_t = (1 - t) * x_0 + t * x_1
        # 其中：
        # - x_0: 干净的数据（pixel_latents）
        # - x_1: 纯噪声（noise）
        # - t: 时间参数（sigmas）
        # - x_t: 时间t的噪声数据

        # 物理意义：
        # t=0: 完全是干净数据
        # t=1: 完全是噪声
        # 0<t<1: 数据和噪声的线性混合
        ```

        #### 为什么用线性插值？
        ```python
        # 相比传统扩散模型的复杂噪声调度：
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise

        # Flow Matching 更简单：
        # x_t = (1 - t) * x_0 + t * noise

        # 优势：
        # 1. 数学简洁：容易理解和实现
        # 2. 训练稳定：避免复杂的噪声调度
        # 3. 采样高效：ODE求解器可以更好地处理
        ```

        ### 标准化操作

        ```python
        pixel_latents = (pixel_latents - latents_mean) * latents_std
        ```

        #### 为什么要标准化？
        ```python
        # VAE 编码器的输出可能有偏移和缩放
        # 例如：输出范围可能是 [-5, 5] 而不是 [-1, 1]

        # 标准化的目的：
        # 1. 数值稳定：将数据缩放到合适范围
        # 2. 模型兼容：与预训练模型的期望输入匹配
        # 3. 训练效率：标准化的数据更容易优化
        ```

        #### 标准化 vs 归一化
        ```python
        # 标准化（Standardization）：
        # z = (x - mean) / std
        # 结果：均值为0，标准差为1

        # 归一化（Normalization）：
        # z = (x - min) / (max - min)
        # 结果：范围在[0, 1]之间

        # 本例使用的是修改版标准化：
        # z = (x - mean) * (1/std)
        # 这里 latents_std 实际是 1/std
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 5. 损失计算 - 训练目标的精确定义

        ### 均方误差损失

        ```python
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        ```

        #### 为什么要转换为 float()？
        ```python
        # 混合精度训练中的问题：
        # model_pred: torch.float16
        # target: torch.float16
        # weighting: torch.float16

        # 计算 (a - b) ** 2 时，float16 可能溢出
        # 例如：如果 a-b = 300，那么 300^2 = 90000 > 65504（float16最大值）

        # 解决方案：转换为 float32 计算
        loss_float32 = (model_pred.float() - target.float()) ** 2
        # 确保数值稳定性，避免溢出
        ```

        #### reshape() 的作用
        ```python
        # 原始形状：(batch, time, channels, height, width)
        # 例如：(2, 1, 4, 64, 64)

        # reshape 后：(batch, -1)
        # 例如：(2, 16384)  # 1*4*64*64 = 16384

        # 目的：
        # 1. 将每个样本的所有像素展平
        # 2. 便于计算每个样本的平均损失
        # 3. 保持 batch 维度用于后续聚合
        ```

        #### 两次 mean() 的含义
        ```python
        # 第一次 mean(dim=1)：
        # 输入：(batch, num_pixels)
        # 输出：(batch,)
        # 含义：计算每个样本的平均像素损失

        # 第二次 mean()：
        # 输入：(batch,)
        # 输出：标量
        # 含义：计算整个批次的平均损失

        # 为什么分两步？
        # 1. 灵活性：可以在样本级别应用不同权重
        # 2. 数值稳定：避免一次性处理过大的张量
        # 3. 调试方便：可以检查每个样本的损失
        ```

        ### Flow Matching 的目标函数

        ```python
        target = noise - pixel_latents
        ```

        #### 为什么目标是 noise - pixel_latents？
        ```python
        # Flow Matching 的速度场：
        # v_t = d(x_t)/dt = x_1 - x_0 = noise - pixel_latents

        # 物理解释：
        # - 我们想学习从 x_0 到 x_1 的"流动方向"
        # - 在每个时间点，模型预测应该朝哪个方向"流动"
        # - 正确的方向就是 noise - pixel_latents

        # 训练目标：
        # 让模型预测的速度场 model_pred 接近真实速度场 target
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 6. 优化器和参数管理 - 高效训练的关键

        ### 优化器选择

        ```python
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        ```

        #### 为什么选择 AdamW？
        ```python
        # AdamW vs Adam 的区别：
        # Adam: 权重衰减应用在梯度上
        # AdamW: 权重衰减直接应用在权重上

        # AdamW 的优势：
        # 1. 更好的泛化性能
        # 2. 权重衰减与学习率解耦
        # 3. 在大模型训练中表现更稳定
        ```

        #### 参数详解
        ```python
        # lr: 学习率
        # - 控制参数更新的步长
        # - LoRA 通常需要比全量微调更高的学习率

        # betas: 动量参数
        # - beta1: 一阶矩估计的衰减率（通常0.9）
        # - beta2: 二阶矩估计的衰减率（通常0.999）

        # weight_decay: 权重衰减
        # - L2正则化，防止过拟合
        # - 对LoRA特别重要，因为参数相对较少

        # eps: 数值稳定性参数
        # - 防止除零错误
        # - 通常设为1e-8
        ```

        ### 参数过滤和统计

        ```python
        for n, param in flux_transformer.named_parameters():
            if 'lora' not in n:
                param.requires_grad = False
            else:
                param.requires_grad = True

        print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
        ```

        #### 参数统计的重要性
        ```python
        # 为什么要统计可训练参数？
        # 1. 资源估算：预估显存和计算需求
        # 2. 效率验证：确认LoRA确实减少了参数量
        # 3. 调试帮助：检查参数冻结是否正确

        # numel() 的作用：
        # 返回张量中元素的总数
        # 例如：(1024, 512) 的矩阵有 1024 * 512 = 524288 个参数
        ```

        #### 参数过滤的策略
        ```python
        # 只训练包含 'lora' 的参数
        lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())

        # 为什么要过滤？
        # 1. 内存效率：优化器只为可训练参数分配状态
        # 2. 计算效率：减少优化器的计算开销
        # 3. 正确性保证：确保只更新想要训练的参数
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 7. 梯度控制和无梯度上下文 - 精确的计算控制

        ### torch.no_grad() 上下文管理器

        ```python
        with torch.no_grad():
            pixel_latents = vae.encode(pixel_values).latent_dist.sample()
            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(...)
        ```

        #### 为什么要使用 no_grad()？
        ```python
        # 问题场景：VAE编码和文本编码不需要训练
        # 如果不使用 no_grad()：
        # 1. PyTorch 会为这些操作构建计算图
        # 2. 占用大量显存存储中间梯度
        # 3. 增加不必要的计算开销

        # 使用 no_grad() 的效果：
        # 1. 禁用自动微分
        # 2. 节省显存
        # 3. 提高计算速度
        # 4. 明确表示这些操作不参与训练
        ```

        #### no_grad() vs detach()
        ```python
        # no_grad() - 上下文管理器
        with torch.no_grad():
            result = some_operation(input)  # 整个操作都不计算梯度

        # detach() - 张量方法
        result = some_operation(input).detach()  # 只是结果不计算梯度

        # 选择原则：
        # - 大块操作用 no_grad()
        # - 单个张量用 detach()
        ```

        ### requires_grad 属性管理

        ```python
        vae.requires_grad_(False)
        flux_transformer.requires_grad_(False)
        ```

        #### 模型级别的梯度控制
        ```python
        # requires_grad_(False) 的作用：
        # 1. 将模型所有参数的 requires_grad 设为 False
        # 2. 防止意外的梯度计算
        # 3. 明确表示模型不参与训练

        # 与 no_grad() 的区别：
        # requires_grad_(False): 永久性设置，影响参数本身
        # no_grad(): 临时性设置，只影响当前操作
        ```

        #### 精细的梯度控制
        ```python
        # 在LoRA训练中的梯度控制策略：
        # 1. 冻结整个模型
        flux_transformer.requires_grad_(False)

        # 2. 添加LoRA适配器
        flux_transformer.add_adapter(lora_config)

        # 3. 只启用LoRA参数的梯度
        for name, param in flux_transformer.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        # 结果：只有LoRA参数参与训练，其他参数保持冻结
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 8. 设备管理和数据转移 - 分布式训练的基础

        ### 模型设备转移

        ```python
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
        text_encoding_pipeline.to(accelerator.device)
        vae.to(accelerator.device, dtype=weight_dtype)
        ```

        #### 为什么要显式指定设备？
        ```python
        # 分布式训练中的设备分配：
        # 进程0: accelerator.device = cuda:0
        # 进程1: accelerator.device = cuda:1
        # 进程2: accelerator.device = cuda:2
        # 进程3: accelerator.device = cuda:3

        # 如果不指定设备：
        # 1. 模型可能在错误的GPU上
        # 2. 数据和模型设备不匹配
        # 3. 导致运行时错误或性能问题
        ```

        #### .to() 方法的双重作用
        ```python
        # 同时指定设备和数据类型
        model.to(device=accelerator.device, dtype=weight_dtype)

        # 等价于：
        model.to(accelerator.device)  # 转移设备
        model.to(weight_dtype)        # 转换数据类型

        # 一次性操作的优势：
        # 1. 代码简洁
        # 2. 避免多次内存拷贝
        # 3. 确保设备和类型的一致性
        ```

        ### 数据设备一致性

        ```python
        pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
        ```

        #### 链式调用的顺序
        ```python
        # 推荐顺序：先类型转换，再设备转移
        tensor = tensor.to(dtype=target_dtype).to(device=target_device)

        # 原因：
        # 1. 类型转换通常在CPU上更快
        # 2. 减少GPU内存的临时占用
        # 3. 避免不必要的设备间数据传输
        ```

        #### 设备检查的重要性
        ```python
        # 在复杂的训练流程中，经常需要检查设备一致性：
        assert model.device == data.device, f"设备不匹配: {model.device} vs {data.device}"

        # 常见的设备不匹配问题：
        # 1. 数据在CPU，模型在GPU
        # 2. 不同的GPU之间
        # 3. 混合精度训练中的类型不匹配
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 9. 总结：PyTorch 在深度学习训练中的核心作用

        ### 关键操作分类

        #### 1. 数据管理
        ```python
        # 张量创建和形状操作
        torch.randn_like()     # 创建同形状随机张量
        tensor.unsqueeze()     # 添加维度
        tensor.permute()       # 重排维度
        tensor.view()          # 重塑形状
        tensor.reshape()       # 灵活重塑

        # 设备和类型管理
        tensor.to(device, dtype)  # 设备和类型转移
        ```

        #### 2. 数值计算
        ```python
        # 基础运算
        (1.0 - sigmas) * pixel_latents + sigmas * noise  # 线性组合
        (pixel_latents - latents_mean) * latents_std      # 标准化

        # 统计操作
        torch.mean()           # 均值计算
        tensor ** 2            # 平方运算
        ```

        #### 3. 梯度控制
        ```python
        # 梯度管理
        torch.no_grad()        # 禁用自动微分
        param.requires_grad    # 参数梯度控制
        model.requires_grad_() # 模型级梯度控制
        ```

        #### 4. 优化器
        ```python
        # 参数优化
        torch.optim.AdamW      # 优化器选择
        filter(lambda p: p.requires_grad, model.parameters())  # 参数过滤
        ```

        ### 设计原则总结

        #### 1. 数值稳定性优先
        ```python
        # 混合精度训练中的类型转换
        loss_float32 = (model_pred.float() - target.float()) ** 2
        # 确保关键计算使用足够的精度
        ```

        #### 2. 内存效率考虑
        ```python
        # 使用广播而不是显式扩展
        result = tensor - mean.view(1, 1, -1, 1, 1)  # 广播
        # 而不是 tensor - mean.expand_as(tensor)     # 显式扩展
        ```

        #### 3. 设备一致性保证
        ```python
        # 创建张量时直接指定目标设备
        noise = torch.randn_like(data, device=target_device)
        # 避免后续的设备转移开销
        ```

        #### 4. 梯度控制精确性
        ```python
        # 明确区分训练和推理部分
        with torch.no_grad():  # 推理部分
            encoded = vae.encode(images)

        # 训练部分自动计算梯度
        loss = criterion(model_output, target)
        ```

        ### PyTorch 的核心价值

        在深度学习训练中，PyTorch 提供了：

        - **灵活性**：动态计算图，易于调试和实验
        - **效率性**：高度优化的张量操作和GPU加速
        - **可控性**：精确的梯度控制和内存管理
        - **扩展性**：支持分布式训练和混合精度
        - **易用性**：直观的API设计和丰富的功能

        这些特性使得 PyTorch 成为现代深度学习研究和应用的首选框架。
        """
    )
    return


if __name__ == "__main__":
    app.run()
