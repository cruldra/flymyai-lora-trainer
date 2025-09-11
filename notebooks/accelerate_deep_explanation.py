import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(
        r"""
        # Accelerate 库 API 深度解析 - 为什么这么写？

        这个笔记本将深入解释 `train.py` 中每个 Accelerate API 的使用原因、目的和解决的具体问题。

        ## 背景：分布式训练的挑战

        在深度学习训练中，我们经常面临：

        1. **显存不足** - 模型太大，单GPU装不下
        2. **训练太慢** - 数据集很大，单GPU训练时间太长
        3. **批次大小限制** - 想要大批次训练但显存不够

        Accelerate 就是为了解决这些问题而设计的。
        """
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. ProjectConfiguration - 为什么需要项目配置？

        ```python
        accelerator_project_config = ProjectConfiguration(
            project_dir=args.output_dir, 
            logging_dir=logging_dir
        )
        ```

        **问题场景**：
        假设你有4个GPU在训练，每个GPU是一个进程：

        - 进程0在GPU0上运行
        - 进程1在GPU1上运行  
        - 进程2在GPU2上运行
        - 进程3在GPU3上运行

        **如果不用ProjectConfiguration会发生什么？**
        ```python
        # 每个进程可能会：
        os.makedirs("./output")  # 4个进程同时创建目录 -> 冲突！
        torch.save(model, "./model.pt")  # 4个进程同时保存 -> 文件损坏！
        ```

        **ProjectConfiguration的解决方案**：

        - 确保只有主进程（进程0）创建目录和保存文件
        - 其他进程等待主进程完成操作
        - 所有进程使用统一的路径配置
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. Accelerator初始化 - 核心配置的原因

        ```python
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )
        ```

        ### gradient_accumulation_steps - 为什么需要梯度累积？

        **问题**：你想用批次大小64训练，但每个GPU只能装下批次大小16

        **传统做法**：
        ```python
        # 只能用小批次，效果不好
        for batch in dataloader:  # batch_size=16
            loss = model(batch)
            loss.backward()
            optimizer.step()
        ```

        **梯度累积解决方案**：
        ```python
        # 累积4次梯度，等效于批次大小64
        gradient_accumulation_steps = 4
        for i, batch in enumerate(dataloader):  # batch_size=16
            loss = model(batch) / 4  # 除以累积步数
            loss.backward()  # 累积梯度
            if (i + 1) % 4 == 0:  # 每4步更新一次
                optimizer.step()
                optimizer.zero_grad()
        ```

        **Accelerate自动处理**：你只需要设置参数，它自动管理累积逻辑
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### mixed_precision - 为什么需要混合精度？

        **问题**：模型训练显存不够，速度太慢

        **传统float32训练**：
        ```python
        # 每个参数占用4字节
        model = Model().float()  # 所有参数都是float32
        # 一个1B参数的模型需要4GB显存
        ```

        **混合精度训练**：
        ```python
        # 前向传播用float16（2字节），反向传播用float32（4字节）
        model = Model().half()  # 前向传播用float16
        # 显存减少一半，速度提升2倍
        ```

        **但是有问题**：
        - float16精度低，可能导致梯度消失
        - 需要手动处理精度转换

        **Accelerate的解决方案**：

        - 自动在合适的地方使用float16/float32
        - 自动处理梯度缩放，防止梯度消失
        - 你只需要设置 `mixed_precision="fp16"`
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. 设备管理 - 为什么要用accelerator.device？

        ```python
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
        text_encoding_pipeline.to(accelerator.device)
        ```

        **问题场景**：分布式训练中，每个进程需要使用不同的GPU

        **传统做法**：
        ```python
        # 你需要手动管理设备分配
        if torch.cuda.device_count() > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        ```

        **Accelerate的解决方案**：
        ```python
        # 自动为每个进程分配正确的设备
        model.to(accelerator.device)
        # 进程0 -> cuda:0
        # 进程1 -> cuda:1  
        # 进程2 -> cuda:2
        # 进程3 -> cuda:3
        ```

        **为什么这样更好？**

        - 你不需要知道当前进程应该用哪个GPU
        - 代码在单GPU和多GPU环境下都能正常运行
        - 自动处理CPU/GPU/TPU的差异
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 4. 进程检查 - 为什么要区分主进程？

        ```python
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
        ```

        **问题**：在4GPU训练中，如果每个进程都执行相同操作：

        ```python
        # 错误的做法 - 每个进程都执行
        os.makedirs("output")  # 4个进程同时创建 -> 竞争条件
        print("Training started")  # 打印4次相同信息 -> 日志混乱
        wandb.init(project="test")  # 创建4个wandb实验 -> 资源浪费
        ```

        **正确的做法**：
        ```python
        # 只有主进程执行一次性操作
        if accelerator.is_main_process:
            os.makedirs("output")  # 只创建一次
            wandb.init(project="test")  # 只初始化一次
            
        # 每个进程都执行的操作
        model.train()  # 每个进程都需要设置训练模式
        ```

        **is_main_process vs is_local_main_process**：
        
        - `is_main_process`: 全局主进程（多机训练中只有一个）
        - `is_local_main_process`: 本机主进程（每台机器一个）
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 5. accelerator.prepare() - 最关键的魔法

        ```python
        flux_transformer, optimizer, _, lr_scheduler = accelerator.prepare(
            flux_transformer, optimizer, deepcopy(train_dataloader), lr_scheduler
        )
        ```

        **这一行代码做了什么？**

        ### 对模型的包装：
        ```python
        # 原始模型
        model = MyModel()
        
        # prepare后变成
        model = DistributedDataParallel(model)  # 如果是多GPU
        # 或者
        model = DataParallel(model)  # 如果是单机多GPU
        ```

        ### 对优化器的包装：
        ```python
        # 原始优化器
        optimizer = torch.optim.Adam(model.parameters())
        
        # prepare后添加了混合精度支持
        optimizer = 带有GradScaler的优化器  # 自动处理梯度缩放
        ```

        ### 对数据加载器的包装：
        ```python
        # 原始数据加载器
        dataloader = DataLoader(dataset, batch_size=32)
        
        # prepare后变成分布式数据加载器
        dataloader = DistributedDataLoader(dataset)
        # 每个进程只处理部分数据，避免重复
        ```

        **为什么要这样包装？**

        - 你写的代码保持不变
        - Accelerate在后台处理所有分布式训练的复杂性
        - 同一份代码可以在单GPU/多GPU/多机环境运行
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 6. 梯度累积的精妙设计

        ```python
        with accelerator.accumulate(flux_transformer):
            loss = model(batch)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        ```

        **为什么要用accelerator.accumulate()？**

        **传统梯度累积的问题**：
        ```python
        # 手动梯度累积 - 容易出错
        for i, batch in enumerate(dataloader):
            loss = model(batch) / accumulation_steps  # 容易忘记除法
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:  # 容易算错
                optimizer.step()
                optimizer.zero_grad()
        ```

        **Accelerate的解决方案**：
        ```python
        with accelerator.accumulate(model):
            loss = model(batch)  # 不需要手动除法
            accelerator.backward(loss)  # 自动处理缩放
            # 自动判断是否该更新参数
        ```

        **accelerate.accumulate()内部做了什么？**

        1. 自动计算loss缩放（除以累积步数）
        2. 管理梯度同步时机
        3. 在分布式训练中协调所有进程
        4. 设置sync_gradients标志
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 7. accelerator.backward() - 为什么不用loss.backward()？

        ```python
        accelerator.backward(loss)  # 而不是 loss.backward()
        ```

        **传统反向传播的问题**：
        ```python
        # 在混合精度训练中
        loss = model(batch)
        loss.backward()  # 可能导致梯度下溢出（变成0）
        ```

        **混合精度训练的挑战**：

        - float16的数值范围很小（6.1e-5 到 65504）
        - 梯度经常小于6.1e-5，会变成0
        - 导致训练停滞

        **accelerator.backward()的解决方案**：
        ```python
        # 内部实现类似于：
        if mixed_precision:
            # 1. 梯度缩放
            scaled_loss = loss * scale_factor  # 放大loss
            scaled_loss.backward()  # 计算放大的梯度
            # 2. 梯度还原
            for param in model.parameters():
                param.grad = param.grad / scale_factor  # 还原梯度
        else:
            loss.backward()  # 普通反向传播
        ```

        **为什么这样设计？**
        
        - 你的代码保持简单：只需要调用accelerator.backward()
        - Accelerate自动处理混合精度的复杂性
        - 防止梯度下溢出，保证训练稳定性
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 8. accelerator.gather() - 分布式数据收集的必要性

        ```python
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        ```

        **问题场景**：4GPU训练，每个GPU计算一个loss值

        **如果不用gather会怎样？**
        ```python
        # 每个进程只看到自己的loss
        # GPU0: loss = 0.5
        # GPU1: loss = 0.3
        # GPU2: loss = 0.7
        # GPU3: loss = 0.4

        # 如果直接记录
        accelerator.log({"loss": loss})  # 只记录当前进程的loss
        # 结果：日志中只有一个GPU的loss，不代表全局情况
        ```

        **accelerator.gather()的作用**：
        ```python
        # 收集所有进程的loss
        all_losses = accelerator.gather(loss)
        # 结果：[0.5, 0.3, 0.7, 0.4]

        avg_loss = all_losses.mean()  # 0.475
        # 这才是真正的全局平均loss
        ```

        **为什么要repeat(args.train_batch_size)？**
        ```python
        # 每个GPU处理不同数量的样本时
        # GPU0: 16个样本，loss=0.5
        # GPU1: 16个样本，loss=0.3
        # GPU2: 12个样本，loss=0.7  # 最后一个batch可能不满
        # GPU3: 8个样本，loss=0.4

        # 需要按样本数量加权平均
        weighted_avg = (0.5*16 + 0.3*16 + 0.7*12 + 0.4*8) / (16+16+12+8)
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 9. sync_gradients - 梯度同步的时机控制

        ```python
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        ```

        **为什么需要sync_gradients判断？**

        **梯度累积场景**：
        ```python
        gradient_accumulation_steps = 4

        # 步骤1: 计算第1个batch的梯度
        loss1 = model(batch1)
        accelerator.backward(loss1)  # 梯度累积到参数上
        # sync_gradients = False，不更新参数

        # 步骤2: 计算第2个batch的梯度
        loss2 = model(batch2)
        accelerator.backward(loss2)  # 梯度继续累积
        # sync_gradients = False，不更新参数

        # 步骤3: 计算第3个batch的梯度
        loss3 = model(batch3)
        accelerator.backward(loss3)  # 梯度继续累积
        # sync_gradients = False，不更新参数

        # 步骤4: 计算第4个batch的梯度
        loss4 = model(batch4)
        accelerator.backward(loss4)  # 梯度继续累积
        # sync_gradients = True，现在可以更新参数了！
        ```

        **如果没有这个判断会怎样？**
        ```python
        # 错误的做法
        for batch in dataloader:
            loss = model(batch)
            accelerator.backward(loss)
            optimizer.step()  # 每次都更新 -> 梯度累积失效！
            optimizer.zero_grad()
        ```

        **sync_gradients的作用**：

        - 告诉你是否完成了一个完整的训练步骤
        - 只有在梯度累积完成时才更新参数
        - 确保分布式训练中所有进程同步
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 10. accelerator.wait_for_everyone() - 进程同步的重要性

        ```python
        accelerator.wait_for_everyone()
        accelerator.end_training()
        ```

        **为什么需要等待所有进程？**

        **问题场景**：训练结束时的竞争条件
        ```python
        # 没有同步的情况
        # GPU0: 训练完成，开始保存模型
        # GPU1: 还在训练最后几个batch
        # GPU2: 训练完成，也开始保存模型  -> 冲突！
        # GPU3: 还在训练

        # 结果：模型文件可能损坏或不完整
        ```

        **wait_for_everyone()的作用**：
        ```python
        # 确保所有进程都完成训练
        accelerator.wait_for_everyone()

        # 现在所有进程都到达这里，可以安全地保存模型
        if accelerator.is_main_process:
            model.save_pretrained("./final_model")
        ```

        **类似的同步场景**：

        - 保存检查点前：确保所有进程完成当前步骤
        - 验证阶段前：确保所有进程完成训练步骤
        - 程序结束前：确保所有资源正确释放

        **如果不同步会怎样？**
        
        - 文件损坏：多个进程同时写入
        - 数据不一致：某些进程的结果丢失
        - 资源泄漏：某些进程提前退出，GPU内存未释放
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 11. 总结：Accelerate解决的核心问题

        ### 问题1：分布式训练的复杂性
        **传统做法**：
        ```python
        # 需要手动处理的事情：
        torch.distributed.init_process_group()  # 初始化进程组
        model = DistributedDataParallel(model)  # 包装模型
        sampler = DistributedSampler(dataset)   # 分布式采样
        # ... 还有很多细节
        ```

        **Accelerate方案**：
        ```python
        accelerator = Accelerator()
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        # 一行代码搞定所有分布式设置
        ```

        ### 问题2：混合精度训练的陷阱
        **传统做法**：
        ```python
        scaler = GradScaler()  # 手动创建缩放器
        with autocast():       # 手动管理精度
            loss = model(batch)
        scaler.scale(loss).backward()  # 手动缩放
        scaler.step(optimizer)         # 手动更新
        scaler.update()               # 手动更新缩放因子
        ```

        **Accelerate方案**：
        ```python
        accelerator = Accelerator(mixed_precision="fp16")
        # 所有混合精度逻辑自动处理
        accelerator.backward(loss)  # 自动缩放和更新
        ```

        ### 问题3：代码可移植性差
        **传统问题**：

        - 单GPU代码不能直接用于多GPU
        - 多GPU代码不能直接用于多机
        - 需要为不同环境写不同的代码

        **Accelerate解决方案**：

        - 同一份代码在所有环境运行
        - 通过配置文件或环境变量控制行为
        - 开发时用单GPU，部署时用多GPU，无需改代码
        """
    )
    return


if __name__ == "__main__":
    app.run()
