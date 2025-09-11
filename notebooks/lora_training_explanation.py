import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # 🎨 LoRA训练代码详解

        > 通过你的 `train_lora.yaml` 配置文件，详细解释 `train.py` 中的代码流程

        我们会用通俗易懂的语言来解释每个步骤，就像搭积木一样，一步步理解整个训练过程。
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🎯 什么是LoRA训练？

        想象一下，你有一个很厉害的AI画家（预训练模型），但是它不太会画你想要的特定风格。

        **LoRA（Low-Rank Adaptation）** 就像是给这个AI画家戴上一副"特殊眼镜"：

        | 特点 | 说明 |
        |------|------|
        | 🎨 **不改变原技能** | 不修改原模型参数 |
        | 👓 **添加小滤镜** | 只加入LoRA适配器 |
        | 🎯 **学习新风格** | 专门学习你想要的特定风格 |
        | 💾 **体积小巧** | 容易保存和分享 |
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 📋 你的训练配置解读

        让我们先看看你的 `train_lora.yaml` 配置文件，理解训练的基本设置：
        """
    )
    return


@app.cell
def __(mo):
    config_data = [
        {
            "配置项": "pretrained_model_name_or_path",
            "值": "./models/Qwen-Image",
            "说明": "🎨 基础AI画家的位置",
        },
        {"配置项": "img_dir", "值": "./datasets/lol", "说明": "📁 训练图片的文件夹"},
        {
            "配置项": "max_train_steps",
            "值": "3000",
            "说明": "🔄 训练步数（练习3000次）",
        },
        {
            "配置项": "learning_rate",
            "值": "1e-4",
            "说明": "📈 学习速度（温和的学习速度）",
        },
        {"配置项": "rank", "值": "16", "说明": "🎛️ LoRA的复杂度（适中）"},
        {"配置项": "train_batch_size", "值": "1", "说明": "📦 每次处理的图片数量"},
        {"配置项": "img_size", "值": "1024", "说明": "🖼️ 图片尺寸"},
        {"配置项": "mixed_precision", "值": "bf16", "说明": "⚡ 混合精度训练"},
    ]

    mo.ui.table(data=config_data, selection=None, pagination=False)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 🚀 代码流程详解

        ### 第一步：准备工作
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ```python
        def main():
            args = OmegaConf.load(parse_args())  # 读取配置文件
            accelerator = Accelerator(...)       # 设置训练加速器
        ```

        **这一步在做什么？**

        | 步骤 | 作用 | 比喻 |
        |------|------|------|
        | 📖 **读取配置** | 读取 `train_lora.yaml` | 厨师看菜谱 |
        | ⚙️ **设置环境** | 初始化 `Accelerator` | 准备厨房工具 |
        | 🎯 **确定目标** | 知道训练参数 | 明确要做什么菜 |
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 第二步：加载AI模型组件
        """
    )
    return


@app.cell
def __(mo):
    components_data = [
        {
            "组件": "📝 文本理解师",
            "变量名": "text_encoding_pipeline",
            "作用": "理解文字描述",
            "比喻": "翻译官",
        },
        {
            "组件": "🖼️ 图像编码器",
            "变量名": "vae",
            "作用": "压缩图片为latent",
            "比喻": "压缩软件",
        },
        {
            "组件": "🎨 核心画家",
            "变量名": "flux_transformer",
            "作用": "真正的AI画家",
            "比喻": "艺术家",
        },
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            text_encoding_pipeline = QwenImagePipeline.from_pretrained(...)
            vae = AutoencoderKLQwenImage.from_pretrained(...)
            flux_transformer = QwenImageTransformer2DModel.from_pretrained(...)
            ```
            """),
            mo.md(r"""
            **AI画家团队组成：**
            """),
            mo.ui.table(data=components_data, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 第三步：安装LoRA"眼镜"
        """
    )
    return


@app.cell
def __(mo):
    lora_steps = [
        {
            "步骤": "🛠️ 创建配置",
            "代码": "LoraConfig(r=16)",
            "作用": "设置眼镜复杂度",
            "比喻": "选择眼镜度数",
        },
        {
            "步骤": "🎯 选择位置",
            "代码": "target_modules=[...]",
            "作用": "指定安装位置",
            "比喻": "选择镜片位置",
        },
        {
            "步骤": "❄️ 冻结原模型",
            "代码": "requires_grad_(False)",
            "作用": "保护原技能",
            "比喻": "锁定原有能力",
        },
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            lora_config = LoraConfig(
                r=args.rank,                    # rank=16 (你配置的)
                target_modules=["to_k", "to_q", "to_v", "to_out.0"]
            )
            flux_transformer.add_adapter(lora_config)
            flux_transformer.requires_grad_(False)  # 冻结原模型
            ```
            """),
            mo.md(r"""
            **LoRA安装过程：**
            """),
            mo.ui.table(data=lora_steps, selection=None, pagination=False),
            mo.md(r"""
            > 💡 **关键理解**：rank=16 是复杂度，数字越大能学更多细节，但容易过拟合
            """),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 第四步：准备训练数据
        """
    )
    return


@app.cell
def __(mo):
    data_flow = [
        {
            "步骤": "📁 读取图片",
            "操作": "从文件夹加载",
            "配置": "img_dir: ./datasets/lol",
            "说明": "图片来源",
        },
        {
            "步骤": "📝 读取描述",
            "操作": "加载文本文件",
            "配置": "caption_type: txt",
            "说明": "图片说明",
        },
        {
            "步骤": "🔄 调整尺寸",
            "操作": "resize图片",
            "配置": "img_size: 1024",
            "说明": "统一大小",
        },
        {
            "步骤": "📦 打包批次",
            "操作": "组成batch",
            "配置": "train_batch_size: 1",
            "说明": "每次1张图",
        },
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            train_dataloader = loader(**args.data_config)
            ```
            """),
            mo.md(r"""
            **数据处理流程：**
            """),
            mo.ui.table(data=data_flow, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 第五步：设置训练器
        """
    )
    return


@app.cell
def __(mo):
    trainer_components = [
        {
            "组件": "🎯 优化器 AdamW",
            "作用": "调整参数",
            "配置值": "lr=1e-4",
            "比喻": "智能教练",
        },
        {
            "组件": "📈 学习率调度器",
            "作用": "控制学习速度",
            "配置值": "constant",
            "比喻": "速度控制器",
        },
        {
            "组件": "💾 梯度检查点",
            "作用": "节省显存",
            "配置值": "enable",
            "比喻": "内存管理器",
        },
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            optimizer = torch.optim.AdamW(lora_layers, lr=args.learning_rate, ...)
            lr_scheduler = get_scheduler(...)
            flux_transformer.enable_gradient_checkpointing()
            ```
            """),
            mo.md(r"""
            **训练器组件：**
            """),
            mo.ui.table(data=trainer_components, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 🔄 核心训练循环

        这是最重要的部分！让我们详细解释训练的每一步：
        """
    )
    return


@app.cell
def __(mo):
    training_steps = [
        {
            "步骤": "📸 获取素材",
            "操作": "img, prompts = batch",
            "目的": "拿到图片和描述",
        },
        {"步骤": "🔄 编码图片", "操作": "vae.encode()", "目的": "压缩为latent"},
        {"步骤": "📝 编码文字", "操作": "encode_prompt()", "目的": "转换为向量"},
        {"步骤": "🌪️ 添加噪声", "操作": "noise + latent", "目的": "制造训练问题"},
        {"步骤": "🤖 AI预测", "操作": "flux_transformer()", "目的": "尝试去噪"},
        {
            "步骤": "📊 计算损失",
            "操作": "loss = (pred - target)²",
            "目的": "评估预测质量",
        },
        {"步骤": "🔧 更新参数", "操作": "optimizer.step()", "目的": "改进LoRA参数"},
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            for step, batch in enumerate(train_dataloader):
                img, prompts = batch  # 获取图片和描述

                # 1. 编码图片为latent
                pixel_latents = vae.encode(pixel_values).latent_dist.sample()

                # 2. 编码文字描述
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(prompts)

                # 3. 添加噪声
                noise = torch.randn_like(pixel_latents)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # 4. AI预测
                model_pred = flux_transformer(noisy_model_input, ...)

                # 5. 计算损失
                target = noise - pixel_latents
                loss = torch.mean((model_pred - target) ** 2)

                # 6. 反向传播
                accelerator.backward(loss)
                optimizer.step()
            ```
            """),
            mo.md(r"""
            **训练循环步骤：**
            """),
            mo.ui.table(data=training_steps, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🧮 训练的数学原理（简化版）

        ### Flow Matching 训练方法：
        """
    )
    return


@app.cell
def __(mo):
    math_explanation = [
        {
            "步骤": "🎭 制造问题",
            "公式": "(1-σ)×图片 + σ×噪声",
            "简单解释": "故意弄脏图片",
            "比喻": "把画弄脏",
        },
        {
            "步骤": "🤖 AI预测",
            "公式": "transformer(脏图片)",
            "简单解释": "AI尝试清理",
            "比喻": "AI当清洁工",
        },
        {
            "步骤": "✅ 检查答案",
            "公式": "目标 = 噪声 - 原图",
            "简单解释": "计算标准答案",
            "比喻": "准备答案纸",
        },
        {
            "步骤": "📊 计算差距",
            "公式": "(预测 - 目标)²",
            "简单解释": "评估预测质量",
            "比喻": "打分评估",
        },
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            # 1. 添加噪声
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            # 2. AI预测
            model_pred = flux_transformer(noisy_model_input, ...)

            # 3. 计算目标
            target = noise - pixel_latents

            # 4. 计算损失
            loss = torch.mean((model_pred - target) ** 2)
            ```
            """),
            mo.md(r"""
            **数学原理解释：**
            """),
            mo.ui.table(data=math_explanation, selection=None, pagination=False),
            mo.md(r"""
            > 💡 **核心思想**：让AI学会从噪声中恢复原图，这样就能生成新图片了！
            """),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 💾 检查点保存机制
        """
    )
    return


@app.cell
def __(mo):
    checkpoint_config = [
        {"配置项": "checkpointing_steps", "值": "250", "说明": "每250步保存一次"},
        {"配置项": "checkpoints_total_limit", "值": "10", "说明": "最多保留10个检查点"},
        {"配置项": "output_dir", "值": "./output", "说明": "保存位置"},
    ]

    checkpoint_benefits = [
        {"好处": "🛡️ 防止意外", "说明": "断电崩溃可恢复", "比喻": "存档游戏进度"},
        {"好处": "🏆 选择最佳", "说明": "回到效果最好的点", "比喻": "选择最佳照片"},
        {"好处": "💽 节省空间", "说明": "只保存LoRA部分", "比喻": "只存重要文件"},
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            if global_step % args.checkpointing_steps == 0:
                # 保存LoRA权重
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                QwenImagePipeline.save_lora_weights(save_path, lora_state_dict)
            ```
            """),
            mo.md(r"""
            **保存配置：**
            """),
            mo.ui.table(data=checkpoint_config, selection=None, pagination=False),
            mo.md(r"""
            **保存的好处：**
            """),
            mo.ui.table(data=checkpoint_benefits, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 📊 训练监控和日志
        """
    )
    return


@app.cell
def __(mo):
    monitoring_items = [
        {
            "监控项": "📈 进度条",
            "显示内容": "当前步数/总步数",
            "意义": "训练进度 (0/3000)",
        },
        {"监控项": "📉 损失值", "显示内容": "train_loss", "意义": "AI预测准确度"},
        {"监控项": "🎛️ 学习率", "显示内容": "lr", "意义": "当前学习速度"},
    ]

    loss_interpretation = [
        {"Loss趋势": "📉 持续下降", "含义": "训练正常", "建议": "继续训练"},
        {"Loss趋势": "📈 不再下降", "含义": "可能过拟合", "建议": "调整参数或停止"},
        {"Loss趋势": "🌊 剧烈波动", "含义": "学习率太高", "建议": "降低学习率"},
    ]

    mo.vstack(
        [
            mo.md(r"""
            ```python
            accelerator.log({"train_loss": train_loss}, step=global_step)
            progress_bar.set_postfix(**logs)
            ```
            """),
            mo.md(r"""
            **监控指标：**
            """),
            mo.ui.table(data=monitoring_items, selection=None, pagination=False),
            mo.md(r"""
            **Loss值解读：**
            """),
            mo.ui.table(data=loss_interpretation, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 🎯 整个流程总结
        """
    )
    return


@app.cell
def __(mo):
    key_points = [
        {"关键要点": "只训练LoRA部分", "说明": "原模型保持不变，节省资源"},
        {"关键要点": "小批量训练", "说明": "每次处理少量数据，稳定训练"},
        {"关键要点": "定期保存", "说明": "防止训练成果丢失"},
        {"关键要点": "监控进度", "说明": "通过loss值判断效果"},
    ]

    mo.vstack(
        [
            mo.md(r"""
            **训练流程图：**
            """),
            mo.mermaid(r"""
            graph TD
                A[📋 读取配置文件] --> B[🧠 加载预训练模型]
                B --> C[🔧 安装LoRA适配器]
                C --> D[📚 准备训练数据]
                D --> E[🏃‍♂️ 设置优化器]
                E --> F[🔄 开始训练循环]
                F --> G[📸 处理一批数据]
                G --> H[🧮 计算损失]
                H --> I[🔧 更新LoRA参数]
                I --> J{💾 保存检查点?}
                J -->|是| K[💾 保存模型]
                J -->|否| L[➡️ 继续下一批]
                K --> L
                L --> M{🏁 达到最大步数?}
                M -->|否| G
                M -->|是| N[🎉 训练完成]
            """),
            mo.md(r"""
            **关键要点：**
            """),
            mo.ui.table(data=key_points, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🛠️ 调优建议和常见问题
        """
    )
    return


@app.cell
def __(mo):
    tuning_params = [
        {
            "参数": "📊 学习率",
            "当前值": "1e-4",
            "调高效果": "训练快但可能不稳定",
            "调低效果": "训练慢但更稳定",
        },
        {
            "参数": "🎛️ LoRA rank",
            "当前值": "16",
            "调高效果": "学更多细节，易过拟合",
            "调低效果": "更稳定，细节可能不够",
        },
        {
            "参数": "⏰ 训练步数",
            "当前值": "3000",
            "调高效果": "学得更充分",
            "调低效果": "训练时间短",
        },
        {
            "参数": "📦 批次大小",
            "当前值": "1",
            "调高效果": "训练快，显存需求高",
            "调低效果": "训练慢，显存需求低",
        },
    ]

    data_quality_tips = [
        {"检查项": "🖼️ 图片质量", "要求": "清晰、高分辨率", "影响": "直接影响生成质量"},
        {"检查项": "📝 描述准确性", "要求": "与图片内容匹配", "影响": "影响文本理解"},
        {"检查项": "📁 数据数量", "要求": "足够多样化", "影响": "影响泛化能力"},
        {"检查项": "🎯 风格一致性", "要求": "目标风格统一", "影响": "影响学习效果"},
    ]

    mo.vstack(
        [
            mo.md(r"""
            **参数调优指南：**
            """),
            mo.ui.table(data=tuning_params, selection=None, pagination=False),
            mo.md(r"""
            **数据质量检查：**
            """),
            mo.ui.table(data=data_quality_tips, selection=None, pagination=False),
            mo.md(r"""
            ### 🎉 训练完成后：

            1. 在 `./output` 文件夹找到训练好的LoRA模型
            2. 可以加载到原模型上使用
            3. 享受你的专属AI画家！
            """),
        ]
    )
    return (
        tuning_params,
        data_quality_tips,
    )


if __name__ == "__main__":
    app.run()
