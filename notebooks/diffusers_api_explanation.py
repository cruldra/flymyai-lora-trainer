import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(
        r"""
        # Diffusers 库 API 深度解析 - 扩散模型训练

        这个笔记本将深入解释 `train.py` 中使用的 Diffusers 库的各种 API，
        重点讲解扩散模型训练的核心组件和工作原理。

        ## 什么是 Diffusers？

        Diffusers 是 Hugging Face 开发的扩散模型库，提供了：
        
        - **预训练模型**：DALL-E、Stable Diffusion、Flux 等
        - **训练工具**：噪声调度器、损失函数、优化工具
        - **推理管道**：完整的图像生成流程
        - **模型组件**：VAE、UNet、Transformer 等

        ## 扩散模型基础

        扩散模型通过两个过程工作：
        
        1. **前向过程（加噪）**：逐步向图像添加噪声，直到变成纯噪声
        2. **反向过程（去噪）**：学习从噪声中恢复原始图像

        训练目标：让模型学会预测每一步应该去除多少噪声
        """
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. 核心组件导入

        ```python
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers import (
            AutoencoderKLQwenImage,
            QwenImagePipeline,
            QwenImageTransformer2DModel,
        )
        ```

        ### 组件分工

        #### FlowMatchEulerDiscreteScheduler - 噪声调度器
        **作用**：控制训练和推理过程中的噪声添加和去除
        
        - 定义噪声添加的时间步长
        - 计算每个时间步的噪声强度
        - 提供采样算法（Euler方法）

        #### AutoencoderKLQwenImage - VAE编码器
        **作用**：在像素空间和潜在空间之间转换
        
        - **编码**：图像 → 潜在表示（压缩）
        - **解码**：潜在表示 → 图像（重建）
        - **优势**：在低维空间训练，节省计算资源

        #### QwenImageTransformer2DModel - 主干网络
        **作用**：学习去噪的核心模型
        
        - 接收噪声潜在表示和文本条件
        - 预测应该去除的噪声
        - 基于Transformer架构，处理序列化的图像块

        #### QwenImagePipeline - 推理管道
        **作用**：整合所有组件，提供完整的生成流程
        
        - 文本编码
        - 噪声采样
        - 迭代去噪
        - 图像解码
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. 模型加载 - from_pretrained()

        ```python
        text_encoding_pipeline = QwenImagePipeline.from_pretrained(
            args.pretrained_model_name_or_path, 
            transformer=None, 
            vae=None, 
            torch_dtype=weight_dtype
        )
        
        vae = AutoencoderKLQwenImage.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )
        
        flux_transformer = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
        )
        ```

        ### 为什么分别加载组件？

        #### 训练需求的差异
        ```python
        # 在训练中，我们需要：
        # 1. 冻结 VAE（不训练）
        vae.requires_grad_(False)
        
        # 2. 只训练 Transformer 的 LoRA 部分
        flux_transformer.add_adapter(lora_config)
        
        # 3. 使用 Pipeline 进行文本编码（不训练）
        text_encoding_pipeline.to(accelerator.device)
        ```

        #### 内存优化
        ```python
        # Pipeline 加载时排除大组件
        text_encoding_pipeline = QwenImagePipeline.from_pretrained(
            model_path,
            transformer=None,  # 不加载transformer，节省内存
            vae=None,         # 不加载vae，节省内存
        )
        # 只保留文本编码器和tokenizer
        ```

        #### subfolder 参数的作用
        ```python
        # 预训练模型的目录结构：
        # model_path/
        # ├── vae/
        # │   ├── config.json
        # │   └── diffusion_pytorch_model.safetensors
        # ├── transformer/
        # │   ├── config.json
        # │   └── diffusion_pytorch_model.safetensors
        # └── scheduler/
        #     └── scheduler_config.json
        
        # 通过 subfolder 加载特定组件
        vae = AutoencoderKLQwenImage.from_pretrained(path, subfolder="vae")
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. 噪声调度器 - FlowMatchEulerDiscreteScheduler

        ```python
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        ```

        ### Flow Matching vs 传统扩散

        #### 传统扩散模型（DDPM）
        ```python
        # 噪声添加过程：
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        # 需要复杂的噪声调度表
        ```

        #### Flow Matching（更简单）
        ```python
        # 线性插值过程：
        # x_t = (1 - t) * x_0 + t * noise
        # 其中 t 从 0 到 1
        # 更直观，更容易理解和实现
        ```

        ### Euler 采样方法
        ```python
        # Euler 方法是一种数值积分方法
        # 用于求解常微分方程（ODE）
        
        def euler_step(x_t, model_output, timestep, dt):
            # x_{t+dt} = x_t + dt * model_output
            return x_t + dt * model_output
        ```

        ### 调度器的核心属性
        ```python
        # 时间步长
        timesteps = noise_scheduler.timesteps  # [1000, 999, ..., 1, 0]
        
        # 噪声强度（sigma值）
        sigmas = noise_scheduler.sigmas  # 对应每个时间步的噪声强度
        
        # 训练时间步数
        num_train_timesteps = noise_scheduler.config.num_train_timesteps  # 1000
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 4. VAE 编码和解码

        ```python
        # 编码：图像 → 潜在表示
        pixel_latents = vae.encode(pixel_values).latent_dist.sample()
        
        # 标准化处理
        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = torch.tensor(vae.config.latents_std)
        pixel_latents = (pixel_latents - latents_mean) * latents_std
        ```

        ### 为什么要用 VAE？

        #### 计算效率
        ```python
        # 原始图像：512x512x3 = 786,432 像素
        # VAE潜在表示：64x64x4 = 16,384 元素
        # 压缩比：786,432 / 16,384 = 48倍
        
        # 训练时间对比：
        # 像素空间训练：需要处理78万个值
        # 潜在空间训练：只需处理1.6万个值
        ```

        #### 感知质量
        ```python
        # VAE 训练目标：重建感知上相似的图像
        # 而不是像素级完全一致
        # 这更符合人类视觉感知
        ```

        ### latent_dist.sample() 的作用
        ```python
        # VAE 编码器输出分布参数（均值和方差）
        latent_dist = vae.encode(image)  # 返回分布对象
        
        # 从分布中采样
        latents = latent_dist.sample()  # 随机采样
        # 或者
        latents = latent_dist.mean      # 使用均值（确定性）
        
        # 为什么采样？增加训练的随机性，提高泛化能力
        ```

        ### 标准化的必要性
        ```python
        # VAE 输出的潜在表示可能有偏移和缩放
        # 需要标准化到合适的范围
        normalized_latents = (latents - mean) / std
        
        # 这确保：
        # 1. 数值稳定性
        # 2. 与预训练模型兼容
        # 3. 梯度流动更好
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 5. 时间步采样和噪声添加

        ```python
        # 密度采样
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        
        # 转换为时间步
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices]
        
        # 获取噪声强度
        sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
        
        # 添加噪声
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        ```

        ### 为什么要随机采样时间步？

        #### 训练稳定性
        ```python
        # 如果按顺序训练：
        # 第1个epoch：只训练t=0的去噪
        # 第2个epoch：只训练t=1的去噪
        # ...
        # 问题：模型无法学习不同噪声水平之间的关系
        
        # 随机采样：
        # 每个batch包含不同时间步的样本
        # 模型同时学习所有噪声水平的去噪
        ```

        #### compute_density_for_timestep_sampling 的作用
        ```python
        # 不是均匀采样时间步，而是根据重要性采样
        # 某些时间步对训练更重要，应该被更频繁地采样
        
        # weighting_scheme="none": 均匀采样
        # weighting_scheme="logit_normal": 重点采样中间时间步
        # weighting_scheme="mode": 重点采样特定模式
        ```

        ### Flow Matching 的噪声公式
        ```python
        # Flow Matching 使用线性插值：
        # x_t = (1 - t) * x_clean + t * noise
        # 其中：
        # - t=0: 完全是干净图像
        # - t=1: 完全是噪声
        # - 0<t<1: 部分噪声的混合
        
        # 对应代码：
        # (1.0 - sigmas) * pixel_latents + sigmas * noise
        # sigmas 就是时间步 t 的函数
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 6. 数据打包和文本编码

        ```python
        # 打包潜在表示
        packed_noisy_model_input = QwenImagePipeline._pack_latents(
            noisy_model_input,
            bsz,
            noisy_model_input.shape[2],
            noisy_model_input.shape[3],
            noisy_model_input.shape[4],
        )

        # 文本编码
        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
            prompt=prompts,
            device=packed_noisy_model_input.device,
            num_images_per_prompt=1,
            max_sequence_length=1024,
        )
        ```

        ### _pack_latents() 的作用

        #### 为什么要打包？
        ```python
        # Transformer 模型期望序列输入
        # 但图像是2D的，需要转换为1D序列

        # 原始形状：(batch, channels, height, width)
        # 例如：(4, 16, 64, 64)

        # 打包后形状：(batch, sequence_length, hidden_dim)
        # 例如：(4, 1024, 3072)
        # 其中 1024 = (64/patch_size)^2，3072 = channels * patch_size^2
        ```

        #### 图像分块处理
        ```python
        # 类似 Vision Transformer 的处理方式：
        # 1. 将图像分割成小块（patches）
        # 2. 每个块展平为向量
        # 3. 添加位置编码
        # 4. 输入到 Transformer

        # 例如：64x64的图像，8x8的patch
        # 分成 (64/8)^2 = 64 个patches
        # 每个patch是 8*8*channels 维的向量
        ```

        ### encode_prompt() 的工作流程

        #### 文本处理管道
        ```python
        # 1. 分词（Tokenization）
        tokens = tokenizer(prompts)  # 文本 → token IDs

        # 2. 文本编码
        text_embeddings = text_encoder(tokens)  # tokens → 语义向量

        # 3. 填充和掩码
        # 不同长度的文本需要填充到相同长度
        # mask 标记哪些位置是真实文本，哪些是填充
        ```

        #### 返回值解析
        ```python
        prompt_embeds: torch.Tensor  # 形状: (batch, max_length, hidden_dim)
        # 包含每个token的语义表示

        prompt_embeds_mask: torch.Tensor  # 形状: (batch, max_length)
        # 1表示真实token，0表示填充token
        # 用于注意力机制中忽略填充位置
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 7. Transformer 前向传播

        ```python
        model_pred = flux_transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        ```

        ### 参数详解

        #### hidden_states - 图像输入
        ```python
        # 这是加了噪声的图像潜在表示
        # 模型的任务：预测应该去除的噪声
        # 形状：(batch, sequence_length, hidden_dim)
        ```

        #### timestep - 时间条件
        ```python
        # 告诉模型当前的噪声水平
        # 除以1000是为了归一化到[0,1]范围
        # 不同时间步需要不同的去噪策略

        # t接近0：噪声很少，需要精细去噪
        # t接近1：噪声很多，需要大幅去噪
        ```

        #### encoder_hidden_states - 文本条件
        ```python
        # 文本提示的语义表示
        # 通过交叉注意力机制影响图像生成
        # 形状：(batch, text_length, text_hidden_dim)
        ```

        #### encoder_hidden_states_mask - 注意力掩码
        ```python
        # 防止模型关注填充的token
        # 在交叉注意力计算中：
        # attention_scores[mask == 0] = -inf
        # 确保填充位置的注意力权重为0
        ```

        #### img_shapes - 图像形状信息
        ```python
        # 用于位置编码（RoPE - Rotary Position Embedding）
        # 告诉模型每个patch在原始图像中的位置
        # 对于生成高质量图像很重要
        ```

        ### Transformer 的工作原理
        ```python
        # 1. 自注意力：图像patches之间的关系
        # 2. 交叉注意力：文本和图像之间的关系
        # 3. 前馈网络：非线性变换
        # 4. 输出：预测的噪声
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 8. 数据解包和损失计算

        ```python
        # 解包预测结果
        model_pred = QwenImagePipeline._unpack_latents(
            model_pred,
            height=noisy_model_input.shape[3] * vae_scale_factor,
            width=noisy_model_input.shape[4] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        # 计算损失权重
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

        # Flow Matching 损失
        target = noise - pixel_latents
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        ```

        ### _unpack_latents() 的作用

        #### 序列到图像的转换
        ```python
        # 将 Transformer 输出转换回图像格式
        # 输入：(batch, sequence_length, hidden_dim)
        # 输出：(batch, channels, height, width)

        # 这是 _pack_latents() 的逆操作
        # 将序列化的patches重新组装成2D图像
        ```

        #### vae_scale_factor 的作用
        ```python
        # VAE 的下采样倍数
        # 例如：原图512x512，VAE输出64x64，scale_factor=8

        # 用于计算正确的输出尺寸：
        # output_height = latent_height * vae_scale_factor
        # output_width = latent_width * vae_scale_factor
        ```

        ### Flow Matching 损失函数

        #### 为什么是 noise - pixel_latents？
        ```python
        # Flow Matching 的目标：
        # 学习从 x_t 到 x_0 的"速度场"

        # x_t = (1-t) * x_0 + t * noise
        # 速度场 v_t = d(x_t)/dt = noise - x_0

        # 因此目标就是：noise - pixel_latents
        # 模型学习预测这个速度场
        ```

        #### 损失权重的作用
        ```python
        # 不同时间步的损失可能需要不同的权重
        # weighting_scheme="none": 所有时间步权重相等
        # weighting_scheme="snr": 根据信噪比调整权重
        # weighting_scheme="min_snr": 限制权重范围

        # 目的：平衡不同时间步的学习难度
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 9. 学习率调度器

        ```python
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
        ```

        ### 为什么需要学习率调度？

        #### 训练稳定性
        ```python
        # 扩散模型训练的挑战：
        # 1. 多个时间步的联合优化
        # 2. 文本-图像对齐
        # 3. 高维潜在空间

        # 学习率调度帮助：
        # - 开始时快速学习大致方向
        # - 后期精细调整细节
        ```

        #### 常见调度策略
        ```python
        # "linear": 线性衰减
        # lr_t = lr_0 * (1 - t / T)

        # "cosine": 余弦衰减
        # lr_t = lr_0 * 0.5 * (1 + cos(π * t / T))

        # "constant_with_warmup": 预热后保持常数
        # 前N步线性增长，之后保持不变
        ```

        #### 预热（Warmup）的重要性
        ```python
        # 为什么需要预热？
        # 1. 避免初期梯度爆炸
        # 2. 让模型逐渐适应任务
        # 3. 提高训练稳定性

        # 预热过程：
        # lr_t = lr_0 * min(1, t / warmup_steps)
        ```

        ### 分布式训练中的调整
        ```python
        # 为什么乘以 accelerator.num_processes？

        # 单GPU：1000步预热
        # 4GPU：每个GPU看到的数据是1/4
        # 因此需要 1000 * 4 = 4000 步才能看到相同数量的数据

        # 这确保了分布式训练和单GPU训练的一致性
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 10. 模型保存

        ```python
        # 保存 LoRA 权重
        QwenImagePipeline.save_lora_weights(
            save_path,
            flux_transformer_lora_state_dict,
            safe_serialization=True,
        )
        ```

        ### save_lora_weights() 的优势

        #### 标准化格式
        ```python
        # 保存为 Diffusers 标准格式
        # 可以直接用于推理：

        pipeline = QwenImagePipeline.from_pretrained("base_model")
        pipeline.load_lora_weights("path/to/lora_weights")
        image = pipeline("a beautiful landscape")
        ```

        #### safe_serialization 的作用
        ```python
        # safe_serialization=True: 使用 safetensors 格式
        # 优势：
        # 1. 安全：防止恶意代码执行
        # 2. 快速：加载速度更快
        # 3. 内存效率：支持内存映射

        # safe_serialization=False: 使用 pickle 格式
        # 兼容性更好，但安全性较低
        ```

        ### 完整的推理流程
        ```python
        # 1. 加载基础模型
        pipeline = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16
        )

        # 2. 加载训练好的 LoRA
        pipeline.load_lora_weights("./checkpoint-1000")

        # 3. 生成图像
        image = pipeline(
            prompt="a beautiful landscape with mountains and lakes",
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]

        # 4. 保存结果
        image.save("generated_image.png")
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 11. 训练工具函数

        ```python
        from diffusers.training_utils import (
            compute_density_for_timestep_sampling,
            compute_loss_weighting_for_sd3,
        )
        ```

        ### compute_density_for_timestep_sampling()

        #### 智能时间步采样
        ```python
        # 不是所有时间步都同等重要
        # 某些时间步对最终质量影响更大

        # 采样策略：
        # - 重点采样困难的时间步
        # - 减少简单时间步的采样频率
        # - 提高训练效率
        ```

        #### 参数含义
        ```python
        weighting_scheme="none"     # 均匀采样
        batch_size=bsz             # 批次大小
        logit_mean=0.0             # 分布均值
        logit_std=1.0              # 分布标准差
        mode_scale=1.29            # 模式缩放因子
        ```

        ### compute_loss_weighting_for_sd3()

        #### 损失重加权的必要性
        ```python
        # 问题：不同时间步的损失尺度不同
        # t=0（无噪声）：损失很小，梯度微弱
        # t=1（纯噪声）：损失很大，梯度强烈

        # 解决方案：根据时间步调整损失权重
        # 让所有时间步对训练贡献相等
        ```

        #### 权重计算策略
        ```python
        # "none": 不加权，权重=1
        # "snr": 基于信噪比的权重
        # "min_snr": 限制最小信噪比
        # "snr_trunc": 截断信噪比权重

        # 目标：平衡训练，避免某些时间步主导梯度
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 12. 完整训练流程总结

        ### 数据流向图
        ```python
        # 1. 数据准备
        原始图像 → VAE编码 → 潜在表示 → 标准化
        文本提示 → 分词 → 文本编码 → 语义向量

        # 2. 噪声添加
        潜在表示 + 随机噪声 → 噪声潜在表示

        # 3. 模型预测
        噪声潜在表示 + 时间步 + 文本条件 → Transformer → 预测噪声

        # 4. 损失计算
        预测噪声 vs 真实噪声 → 加权MSE损失

        # 5. 反向传播
        损失 → 梯度 → 更新LoRA参数
        ```

        ### 关键设计原则

        #### 1. 效率优先
        ```python
        # VAE潜在空间：减少计算量
        # LoRA微调：减少参数量
        # 混合精度：减少内存使用
        # 梯度累积：模拟大批次
        ```

        #### 2. 稳定性保证
        ```python
        # 学习率调度：避免训练不稳定
        # 梯度裁剪：防止梯度爆炸
        # 损失重加权：平衡不同时间步
        # 随机时间步：避免过拟合
        ```

        #### 3. 质量控制
        ```python
        # Flow Matching：更简单的训练目标
        # 文本条件：精确控制生成内容
        # 位置编码：保持空间一致性
        # 注意力掩码：忽略无效信息
        ```

        ### 与传统训练的对比

        | 方面 | 传统CNN训练 | 扩散模型训练 |
        |------|-------------|--------------|
        | 输入 | 单张图像 | 图像+噪声+文本 |
        | 目标 | 分类标签 | 噪声预测 |
        | 损失 | 交叉熵 | MSE |
        | 时间维度 | 无 | 多时间步 |
        | 条件控制 | 无 | 文本引导 |
        | 生成能力 | 无 | 强 |

        扩散模型训练比传统训练更复杂，但也更强大，能够生成高质量的图像内容。
        """
    )
    return


if __name__ == "__main__":
    app.run()
