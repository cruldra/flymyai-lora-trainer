import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(
        r"""
        # 工具库深度解析 - 深度学习项目的基础设施

        这个笔记本将深入解释 `train.py` 中使用的非 AI 框架库，
        重点讲解这些工具库在深度学习项目中的作用和重要性。

        ## 工具库的分类

        在深度学习项目中，除了核心的 AI 框架（PyTorch、Transformers、Diffusers 等），
        还需要大量的工具库来支撑整个训练流程：

        - **配置管理**：OmegaConf - 灵活的配置文件处理
        - **命令行解析**：argparse - 标准的参数解析
        - **进度显示**：tqdm - 优雅的进度条
        - **日志记录**：logging - 标准日志系统
        - **文件操作**：os, shutil - 系统级文件管理
        - **对象操作**：copy - 深拷贝和浅拷贝

        ## 为什么需要这些工具库？

        深度学习训练是一个复杂的系统工程，需要：

        1. **可配置性**：不同实验需要不同参数
        2. **可观测性**：了解训练进度和状态
        3. **可维护性**：清晰的日志和错误处理
        4. **可重现性**：精确的配置管理
        5. **可扩展性**：灵活的文件和数据管理
        """
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. OmegaConf - 现代配置管理的艺术

        ```python
        from omegaconf import OmegaConf
        args = OmegaConf.load(parse_args())
        ```

        ### 为什么选择 OmegaConf 而不是传统方法？

        #### 传统配置管理的问题
        ```python
        # 传统方法1：硬编码
        learning_rate = 1e-4
        batch_size = 16
        max_steps = 1000
        # 问题：修改参数需要改代码，不灵活

        # 传统方法2：argparse
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=16)
        # 问题：参数多时命令行变得很长，难以管理

        # 传统方法3：JSON/YAML
        import json
        with open('config.json') as f:
            config = json.load(f)
        # 问题：缺乏类型检查，不支持变量插值
        ```

        #### OmegaConf 的优雅解决方案
        ```python
        # config.yaml
        # model:
        #   name: "qwen-vl"
        #   rank: 64
        # training:
        #   learning_rate: 1e-4
        #   batch_size: 16
        #   max_steps: 1000

        args = OmegaConf.load("config.yaml")
        print(args.model.name)  # "qwen-vl"
        print(args.training.learning_rate)  # 1e-4
        ```

        ### OmegaConf 的核心优势

        #### 1. 层次化配置
        ```python
        # 支持嵌套结构，组织清晰
        args.model.transformer.num_layers
        args.training.optimizer.learning_rate
        args.data.dataset.batch_size
        
        # 而不是扁平的：
        # model_transformer_num_layers
        # training_optimizer_learning_rate
        ```

        #### 2. 类型安全
        ```python
        # OmegaConf 会保持数据类型
        args.training.learning_rate  # float
        args.training.batch_size     # int
        args.model.use_lora         # bool
        
        # 避免了字符串转换的错误
        ```

        #### 3. 变量插值
        ```python
        # config.yaml
        # defaults:
        #   output_dir: "/tmp/experiments"
        # experiment:
        #   name: "lora_rank_64"
        #   full_output_dir: "${defaults.output_dir}/${experiment.name}"
        
        # 结果：full_output_dir = "/tmp/experiments/lora_rank_64"
        ```

        #### 4. 配置合并
        ```python
        # 基础配置
        base_config = OmegaConf.load("base.yaml")
        # 实验特定配置
        exp_config = OmegaConf.load("experiment.yaml")
        # 合并配置
        final_config = OmegaConf.merge(base_config, exp_config)
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. argparse - 命令行接口的标准实现

        ```python
        import argparse

        def parse_args():
            parser = argparse.ArgumentParser(description="Simple example of a training script.")
            parser.add_argument(
                "--config",
                type=str,
                default=None,
                required=True,
                help="path to config",
            )
            args = parser.parse_args()
            return args.config
        ```

        ### 为什么仍然需要 argparse？

        #### 与 OmegaConf 的完美配合
        ```python
        # 命令行只传递配置文件路径
        # python train.py --config experiments/lora_rank_64.yaml
        
        # 而不是传递所有参数
        # python train.py --learning_rate 1e-4 --batch_size 16 --rank 64 ...
        
        # 优势：
        # 1. 命令行简洁
        # 2. 配置可重用
        # 3. 版本控制友好
        ```

        #### argparse 的设计原则
        ```python
        # 1. 描述性帮助信息
        parser = argparse.ArgumentParser(
            description="Simple example of a training script."
        )
        
        # 2. 清晰的参数定义
        parser.add_argument(
            "--config",           # 参数名
            type=str,            # 类型检查
            default=None,        # 默认值
            required=True,       # 必需参数
            help="path to config"  # 帮助信息
        )
        
        # 3. 自动生成帮助
        # python train.py --help
        ```

        #### 错误处理和用户体验
        ```python
        # argparse 自动处理：
        # 1. 参数验证
        if not os.path.exists(args.config):
            parser.error(f"Config file {args.config} not found")
        
        # 2. 帮助信息生成
        # --help 自动显示所有参数说明
        
        # 3. 错误提示
        # 参数错误时给出清晰的错误信息
        ```

        ### 最佳实践：最小化命令行参数
        ```python
        # 好的做法：只传递必要的元参数
        parser.add_argument("--config", required=True)
        parser.add_argument("--output_dir", default="./output")
        parser.add_argument("--resume_from", default=None)
        
        # 避免的做法：传递所有训练参数
        # parser.add_argument("--learning_rate", ...)
        # parser.add_argument("--batch_size", ...)
        # parser.add_argument("--num_epochs", ...)
        # ... 几十个参数
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. tqdm - 进度可视化的优雅实现

        ```python
        from tqdm.auto import tqdm

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not accelerator.is_local_main_process,
        )
        ```

        ### 为什么选择 tqdm.auto？

        #### auto 模块的智能选择
        ```python
        # tqdm.auto 会自动选择最佳的进度条实现：
        # - Jupyter Notebook: 使用 tqdm.notebook (HTML进度条)
        # - 命令行: 使用 tqdm.std (文本进度条)
        # - 其他环境: 自动适配

        from tqdm.auto import tqdm  # 推荐
        # 而不是
        from tqdm import tqdm      # 只适用于命令行
        ```

        #### 分布式训练中的进度条管理
        ```python
        # 问题：4个GPU训练时，每个进程都显示进度条
        # 结果：终端被4个进度条刷屏，难以阅读

        # 解决方案：只在主进程显示
        progress_bar = tqdm(
            range(args.max_train_steps),
            disable=not accelerator.is_local_main_process  # 只有主进程显示
        )

        # 效果：清晰的单一进度条
        ```

        ### tqdm 的高级功能

        #### 1. 动态信息更新
        ```python
        # 在训练循环中更新额外信息
        for step, batch in enumerate(dataloader):
            # ... 训练代码 ...

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            # 显示：Steps: 45%|████▌     | 450/1000 [02:30<02:45, step_loss=0.234, lr=1e-4]
        ```

        #### 2. 嵌套进度条
        ```python
        # 外层：epoch进度
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            # 内层：batch进度
            for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
                # 训练代码
                pass
        ```

        #### 3. 手动控制
        ```python
        # 手动更新进度
        pbar = tqdm(total=total_steps)
        for step in range(total_steps):
            # 训练代码
            pbar.update(1)  # 手动增加1步
            if step % 100 == 0:
                pbar.set_description(f"Loss: {current_loss:.4f}")
        pbar.close()
        ```

        ### 性能考虑
        ```python
        # tqdm 的性能开销很小，但在高频更新时需要注意：

        # 好的做法：适度更新频率
        if step % 10 == 0:  # 每10步更新一次
            progress_bar.set_postfix(loss=loss.item())

        # 避免：每步都更新复杂信息
        # progress_bar.set_postfix(
        #     loss=loss.item(),
        #     grad_norm=compute_grad_norm(),  # 昂贵的计算
        #     memory=get_memory_usage()       # 昂贵的计算
        # )
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 4. logging - 专业的日志管理系统

        ```python
        import logging

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        ```

        ### 为什么需要专业的日志系统？

        #### print() vs logging 的区别
        ```python
        # 使用 print() 的问题：
        print("Training started")
        print(f"Loss: {loss}")
        print("Model saved")

        # 问题：
        # 1. 无法控制输出级别
        # 2. 无时间戳信息
        # 3. 难以重定向到文件
        # 4. 分布式训练时输出混乱
        # 5. 无法区分不同模块的日志
        ```

        #### logging 的专业解决方案
        ```python
        # 结构化的日志信息
        logger.info("Training started")
        logger.debug(f"Batch shape: {batch.shape}")
        logger.warning("Learning rate is very high")
        logger.error("Failed to save checkpoint")

        # 输出格式：
        # 12/25/2024 14:30:15 - INFO - __main__ - Training started
        # 12/25/2024 14:30:16 - DEBUG - __main__ - Batch shape: torch.Size([16, 3, 512, 512])
        ```

        ### 日志级别的智能管理

        #### 分布式训练中的日志控制
        ```python
        # 问题：4个GPU进程都输出日志，信息重复4倍

        # 解决方案：根据进程角色设置不同级别
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            # 非主进程：只显示错误
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # 结果：清晰的单一日志流
        ```

        #### 日志级别的含义
        ```python
        # DEBUG: 详细的调试信息
        logger.debug(f"Processing batch {batch_idx}")

        # INFO: 一般信息
        logger.info("***** Running training *****")

        # WARNING: 警告信息
        logger.warning("Checkpoint directory already exists")

        # ERROR: 错误信息
        logger.error("Failed to load model weights")

        # CRITICAL: 严重错误
        logger.critical("Out of memory, training stopped")
        ```

        ### 日志格式的设计原则
        ```python
        # 标准格式包含关键信息：
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"

        # %(asctime)s: 时间戳 - 追踪事件发生时间
        # %(levelname)s: 日志级别 - 快速识别重要性
        # %(name)s: 记录器名称 - 识别日志来源
        # %(message)s: 实际消息 - 具体信息内容

        # 示例输出：
        # 12/25/2024 14:30:15 - INFO - accelerate.utils - Process 0 initialized
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 5. os 和 shutil - 系统级文件管理

        ```python
        import os
        import shutil

        # 创建输出目录
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)

        # 管理检查点
        checkpoints = os.listdir(args.output_dir)
        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
        shutil.rmtree(removing_checkpoint)
        ```

        ### 为什么需要系统级文件操作？

        #### 深度学习项目的文件管理需求
        ```python
        # 典型的项目目录结构：
        # project/
        # ├── configs/
        # ├── data/
        # ├── outputs/
        # │   ├── checkpoints/
        # │   ├── logs/
        # │   └── results/
        # └── src/

        # 需要动态创建和管理这些目录
        ```

        #### os.makedirs() 的安全创建
        ```python
        # 问题：多进程同时创建目录可能冲突

        # 解决方案1：只在主进程创建
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)

        # exist_ok=True 的作用：
        # - 如果目录已存在，不报错
        # - 如果目录不存在，创建它
        # - 避免竞争条件
        ```

        ### 检查点管理的最佳实践

        #### 自动清理旧检查点
        ```python
        # 问题：检查点文件越来越多，占用大量磁盘空间

        # 解决方案：保留最新的N个检查点
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # 计算需要删除的检查点数量
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                # 安全删除
                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
        ```

        #### 文件操作的安全性考虑
        ```python
        # 1. 路径拼接使用 os.path.join()
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        # 而不是字符串拼接：args.output_dir + "/" + f"checkpoint-{global_step}"

        # 2. 检查文件存在性
        if os.path.exists(checkpoint_path):
            logger.info(f"Checkpoint {checkpoint_path} already exists")

        # 3. 异常处理
        try:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        except OSError as e:
            logger.error(f"Failed to create directory: {e}")
        ```

        ### shutil 的高级文件操作
        ```python
        # shutil.rmtree() - 递归删除目录树
        shutil.rmtree(old_checkpoint_dir)  # 删除整个目录及其内容

        # shutil.copy2() - 复制文件（保留元数据）
        shutil.copy2(source_file, destination_file)

        # shutil.move() - 移动/重命名文件
        shutil.move(temp_checkpoint, final_checkpoint)

        # shutil.disk_usage() - 检查磁盘空间
        total, used, free = shutil.disk_usage(args.output_dir)
        if free < required_space:
            logger.warning("Insufficient disk space")
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 6. copy - 对象复制的精确控制

        ```python
        import copy
        from copy import deepcopy

        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        flux_transformer, optimizer, _, lr_scheduler = accelerator.prepare(
            flux_transformer, optimizer, deepcopy(train_dataloader), lr_scheduler
        )
        ```

        ### 为什么需要对象复制？

        #### 浅拷贝 vs 深拷贝的区别
        ```python
        # 原始对象
        original_scheduler = FlowMatchEulerDiscreteScheduler(...)

        # 浅拷贝 - 只复制对象本身
        shallow_copy = copy.copy(original_scheduler)
        # 问题：内部的张量和配置仍然是共享的

        # 深拷贝 - 递归复制所有内容
        deep_copy = copy.deepcopy(original_scheduler)
        # 优势：完全独立的副本
        ```

        #### 深度学习中的复制需求

        #### 1. 噪声调度器的独立副本
        ```python
        # 为什么需要复制噪声调度器？
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        # 原因：
        # 1. 原始调度器用于模型推理
        # 2. 副本用于训练时的时间步采样
        # 3. 避免训练过程影响原始调度器的状态
        # 4. 确保可重现性
        ```

        #### 2. 数据加载器的复制
        ```python
        # accelerator.prepare() 需要独立的数据加载器副本
        flux_transformer, optimizer, _, lr_scheduler = accelerator.prepare(
            flux_transformer,
            optimizer,
            deepcopy(train_dataloader),  # 深拷贝数据加载器
            lr_scheduler
        )

        # 原因：
        # 1. accelerator.prepare() 会修改数据加载器
        # 2. 分布式训练需要不同的采样策略
        # 3. 保留原始数据加载器用于验证或其他用途
        ```

        ### 复制操作的性能考虑

        #### 何时使用深拷贝
        ```python
        # 需要深拷贝的情况：
        # 1. 对象包含可变的嵌套结构
        config_copy = copy.deepcopy(model_config)

        # 2. 需要完全独立的副本
        scheduler_copy = copy.deepcopy(lr_scheduler)

        # 3. 避免意外的状态共享
        dataloader_copy = copy.deepcopy(dataloader)
        ```

        #### 何时使用浅拷贝
        ```python
        # 适合浅拷贝的情况：
        # 1. 对象主要包含不可变数据
        metadata_copy = copy.copy(metadata_dict)

        # 2. 性能敏感且确认安全
        lightweight_copy = copy.copy(simple_object)
        ```

        #### 性能优化技巧
        ```python
        # 1. 避免不必要的复制
        # 如果对象不会被修改，直接使用引用
        readonly_config = original_config  # 不需要复制

        # 2. 选择性复制
        # 只复制需要修改的部分
        new_config = {
            'model': copy.deepcopy(original_config['model']),
            'data': original_config['data']  # 只读，不需要复制
        }

        # 3. 延迟复制
        # 在真正需要时才进行复制
        def get_scheduler_copy():
            if not hasattr(self, '_scheduler_copy'):
                self._scheduler_copy = copy.deepcopy(self.scheduler)
            return self._scheduler_copy
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 7. 工具库集成的最佳实践

        ### 配置驱动的开发模式

        #### 完整的配置管理流程
        ```python
        # 1. 命令行解析
        config_path = parse_args()  # 只解析配置文件路径

        # 2. 加载配置
        args = OmegaConf.load(config_path)

        # 3. 配置验证
        assert args.training.learning_rate > 0, "Learning rate must be positive"
        assert os.path.exists(args.data.dataset_path), "Dataset path not found"

        # 4. 配置扩展
        args.output_dir = os.path.join(args.base_output_dir, args.experiment_name)
        args.logging_dir = os.path.join(args.output_dir, "logs")
        ```

        #### 环境适配的配置
        ```python
        # 根据运行环境调整配置
        if accelerator.is_main_process:
            # 主进程：完整日志
            logging.getLogger().setLevel(logging.INFO)
        else:
            # 其他进程：只显示错误
            logging.getLogger().setLevel(logging.ERROR)

        # 根据设备调整批次大小
        if torch.cuda.device_count() > 1:
            args.train_batch_size = args.train_batch_size * torch.cuda.device_count()
        ```

        ### 错误处理和恢复机制

        #### 优雅的错误处理
        ```python
        try:
            # 创建输出目录
            os.makedirs(args.output_dir, exist_ok=True)

            # 保存检查点
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        except OSError as e:
            logger.error(f"File operation failed: {e}")
            # 尝试备用路径
            args.output_dir = "/tmp/fallback_output"
            os.makedirs(args.output_dir, exist_ok=True)

        except Exception as e:
            logger.critical(f"Unexpected error: {e}")
            raise
        ```

        #### 资源清理
        ```python
        import atexit

        def cleanup():
            # 程序退出时的清理工作
            logger.info("Cleaning up resources...")

            # 关闭进度条
            if 'progress_bar' in globals():
                progress_bar.close()

            # 清理临时文件
            temp_dir = "/tmp/training_temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # 注册清理函数
        atexit.register(cleanup)
        ```

        ### 性能监控和调试

        #### 集成的监控系统
        ```python
        # 结合多个工具库进行监控
        def monitor_training_step(step, loss, lr):
            # 1. 进度条更新
            progress_bar.set_postfix(
                loss=f"{loss:.4f}",
                lr=f"{lr:.2e}"
            )

            # 2. 日志记录
            if step % args.log_interval == 0:
                logger.info(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")

            # 3. 配置检查
            if step % args.checkpoint_interval == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
        ```

        #### 调试模式的支持
        ```python
        # 根据配置启用调试功能
        if args.debug_mode:
            # 详细日志
            logging.getLogger().setLevel(logging.DEBUG)

            # 更频繁的进度更新
            progress_bar = tqdm(total=args.max_steps, mininterval=0.1)

            # 保留所有检查点
            args.checkpoints_total_limit = None

            logger.debug("Debug mode enabled")
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 8. 总结：工具库的协同价值

        ### 各库的核心职责

        | 库 | 核心职责 | 关键优势 |
        |---|---------|----------|
        | **OmegaConf** | 配置管理 | 层次化、类型安全、变量插值 |
        | **argparse** | 命令行解析 | 标准化、自动帮助、错误处理 |
        | **tqdm** | 进度显示 | 自适应、分布式友好、信息丰富 |
        | **logging** | 日志记录 | 级别控制、格式化、多进程安全 |
        | **os/shutil** | 文件管理 | 跨平台、安全操作、目录管理 |
        | **copy** | 对象复制 | 深浅拷贝、状态隔离、内存控制 |

        ### 集成设计原则

        #### 1. 单一职责原则
        ```python
        # 每个库专注于自己的领域
        config = OmegaConf.load(config_path)    # 配置管理
        logger.info("Training started")         # 日志记录
        progress_bar.update(1)                  # 进度显示
        os.makedirs(output_dir)                 # 文件操作
        ```

        #### 2. 配置驱动原则
        ```python
        # 所有行为都可以通过配置控制
        if config.logging.enable_debug:
            logging.getLogger().setLevel(logging.DEBUG)

        if config.progress.show_bar:
            progress_bar = tqdm(total=config.training.max_steps)
        ```

        #### 3. 环境适配原则
        ```python
        # 根据运行环境自动调整行为
        # Jupyter: 使用 tqdm.notebook
        # 分布式: 只在主进程显示进度和日志
        # 调试: 启用详细输出
        ```

        ### 深度学习项目的基础设施价值

        #### 可重现性保证
        ```python
        # 配置文件 + 版本控制 = 完全可重现的实验
        # config.yaml 记录所有超参数
        # 日志记录训练过程
        # 检查点管理保存模型状态
        ```

        #### 可维护性提升
        ```python
        # 清晰的日志帮助调试
        # 结构化的配置便于修改
        # 自动化的文件管理减少错误
        ```

        #### 可扩展性支持
        ```python
        # 配置系统支持新的实验设置
        # 日志系统支持新的监控指标
        # 文件系统支持新的输出格式
        ```

        ### 最终思考

        这些看似简单的工具库，实际上构成了深度学习项目的**基础设施层**。它们：

        - **降低复杂性**：将系统级操作抽象为简单API
        - **提高可靠性**：提供经过验证的标准实现
        - **增强可观测性**：让训练过程透明可控
        - **支持协作**：标准化的配置和日志便于团队合作

        在深度学习的快速发展中，这些基础工具的稳定性和可靠性，为上层的AI创新提供了坚实的支撑。
        """
    )
    return


if __name__ == "__main__":
    app.run()
