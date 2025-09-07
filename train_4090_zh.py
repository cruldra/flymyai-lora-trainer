# -*- coding: utf-8 -*-
"""
Qwen-Image LoRA 训练脚本 - 内存优化版本
针对显存受限环境（如RTX 4090）进行了多项内存优化
包括：预计算嵌入、模型量化、8位优化器、精细内存管理等
"""

# 标准库导入
import argparse  # 命令行参数解析
import copy  # 对象复制工具
from copy import deepcopy  # 深度复制
import logging  # 日志记录
import os  # 操作系统接口
import shutil  # 高级文件操作

# PyTorch 核心库
import torch  # PyTorch 深度学习框架
from tqdm.auto import tqdm  # 进度条显示

# Accelerate 分布式训练库
from accelerate import Accelerator  # 分布式训练加速器
from accelerate.logging import get_logger  # 分布式日志记录器
from accelerate.utils import ProjectConfiguration  # 项目配置

# 数据集和模型库
import datasets  # HuggingFace 数据集库
import diffusers  # HuggingFace 扩散模型库
from diffusers import FlowMatchEulerDiscreteScheduler  # Flow Matching 调度器
from diffusers import (
    AutoencoderKLQwenImage,  # Qwen 图像 VAE 编码器
    QwenImagePipeline,  # Qwen 图像生成管道
    QwenImageTransformer2DModel,  # Qwen 图像 Transformer 模型
)
from diffusers.optimization import get_scheduler  # 学习率调度器
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,  # 时间步采样密度计算
    compute_loss_weighting_for_sd3,  # SD3 损失权重计算
)
from diffusers.utils import convert_state_dict_to_diffusers  # 状态字典转换
from diffusers.utils.torch_utils import is_compiled_module  # 编译模块检查

# 自定义数据集加载器
from image_datasets.dataset import loader, image_resize  # 图像数据集加载器和调整大小函数

# 配置和模型适配器
from omegaconf import OmegaConf  # YAML/JSON 配置文件解析
from peft import LoraConfig  # LoRA 配置
from peft.utils import get_peft_model_state_dict  # PEFT 模型状态字典获取

# 其他必要库
import transformers  # HuggingFace Transformers 库

# 图像处理库
from PIL import Image  # Python 图像库
import numpy as np  # 数值计算库

# 内存优化相关库
from optimum.quanto import quantize, qfloat8, freeze  # 模型量化库
import bitsandbytes as bnb  # 8位优化器库
from diffusers.loaders import AttnProcsLayers  # 注意力处理器层管理
import gc  # 垃圾回收器

# 初始化日志记录器
logger = get_logger(__name__, log_level="INFO")

# PyTorch 数据集相关（用于测试）
from torch.utils.data import Dataset, DataLoader


def parse_args():
    """
    解析命令行参数
    返回配置文件路径
    """
    parser = argparse.ArgumentParser(description="Qwen-Image LoRA 训练脚本示例")
    parser.add_argument(
        "--config",  # 配置文件参数
        type=str,  # 字符串类型
        default=None,  # 默认值为空
        required=True,  # 必需参数
        help="配置文件路径",  # 帮助信息
    )
    args = parser.parse_args()  # 解析参数
    
    return args.config  # 返回配置文件路径


class ToyDataset(Dataset):
    """
    测试用的玩具数据集
    用于 accelerator.prepare() 的占位符数据集
    """
    def __init__(self, num_samples=100, input_dim=10):
        """
        初始化玩具数据集
        Args:
            num_samples: 样本数量
            input_dim: 输入维度
        """
        self.data = torch.randn(num_samples, input_dim)    # 随机特征数据
        self.labels = torch.randint(0, 2, (num_samples,))  # 随机标签：0或1

    def __getitem__(self, idx):
        """获取单个样本"""
        return self.data[idx], self.labels[idx]

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)


def lora_processors(model):
    """
    递归提取模型中的所有 LoRA 处理器
    这是内存优化的关键：只管理需要训练的 LoRA 层
    
    Args:
        model: 包含 LoRA 适配器的模型
    
    Returns:
        dict: LoRA 处理器字典 {层名称: 模块}
    """
    processors = {}  # 存储 LoRA 处理器的字典

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        """
        递归函数：遍历模型的所有子模块
        
        Args:
            name: 模块名称
            module: 模块对象
            processors: 处理器字典
        """
        if 'lora' in name:  # 如果模块名包含 'lora'
            processors[name] = module  # 添加到处理器字典
            print(name)  # 打印 LoRA 层名称
        
        # 递归遍历所有子模块
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    # 遍历模型的顶级模块
    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def main():
    """主训练函数"""
    # 1. 配置加载和初始化
    args = OmegaConf.load(parse_args())  # 加载 YAML/JSON 配置文件
    logging_dir = os.path.join(args.output_dir, args.logging_dir)  # 日志目录路径

    # 创建 Accelerator 项目配置
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,  # 项目输出目录
        logging_dir=logging_dir  # 日志目录
    )

    # 初始化 Accelerator（分布式训练核心）
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数
        mixed_precision=args.mixed_precision,  # 混合精度训练（fp16/bf16）
        log_with=args.report_to,  # 日志记录工具（wandb/tensorboard等）
        project_config=accelerator_project_config,  # 项目配置
    )
    
    def unwrap_model(model):
        """
        解包模型：从 Accelerator 包装中提取原始模型
        处理编译模块的特殊情况
        """
        model = accelerator.unwrap_model(model)  # 从 Accelerator 包装中解包
        model = model._orig_mod if is_compiled_module(model) else model  # 处理编译模块
        return model

    # 2. 日志配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # 日志格式
        datefmt="%m/%d/%Y %H:%M:%S",  # 日期格式
        level=logging.INFO,  # 日志级别
    )
    
    # 记录 Accelerator 状态
    logger.info(accelerator.state, main_process_only=False)
    
    # 根据进程角色设置不同的日志级别
    if accelerator.is_local_main_process:  # 主进程
        datasets.utils.logging.set_verbosity_warning()  # 数据集日志：警告级别
        transformers.utils.logging.set_verbosity_warning()  # Transformers 日志：警告级别
        diffusers.utils.logging.set_verbosity_info()  # Diffusers 日志：信息级别
    else:  # 非主进程
        datasets.utils.logging.set_verbosity_error()  # 所有库都设为错误级别
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 3. 输出目录创建
    if accelerator.is_main_process:  # 只在主进程中创建目录
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录
    
    # 4. 数据类型设置（根据混合精度配置）
    weight_dtype = torch.float32  # 默认使用 float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16  # 16位浮点数
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16  # Brain Float 16
        args.mixed_precision = accelerator.mixed_precision

    # 5. 文本编码管道初始化（用于预计算文本嵌入）
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,  # 预训练模型路径
        transformer=None,  # 不加载 transformer（节省内存）
        vae=None,  # 不加载 VAE（节省内存）
        torch_dtype=weight_dtype  # 使用指定的数据类型
    )
    text_encoding_pipeline.to(accelerator.device)  # 移动到计算设备

    # 初始化缓存变量
    cached_text_embeddings = None  # 内存中的文本嵌入缓存
    txt_cache_dir = None  # 磁盘文本嵌入缓存目录

    # 6. 缓存目录创建（如果需要预计算）
    if args.precompute_text_embeddings or args.precompute_image_embeddings:
        if accelerator.is_main_process:  # 只在主进程中创建
            cache_dir = os.path.join(args.output_dir, "cache")  # 缓存根目录
            os.makedirs(cache_dir, exist_ok=True)  # 创建缓存目录

    # 7. 文本嵌入预计算（内存优化关键步骤1）
    if args.precompute_text_embeddings:
        with torch.no_grad():  # 禁用梯度计算（节省内存）
            # 选择缓存策略：磁盘 vs 内存
            if args.save_cache_on_disk:
                txt_cache_dir = os.path.join(cache_dir, "text_embs")  # 文本嵌入磁盘缓存目录
                os.makedirs(txt_cache_dir, exist_ok=True)  # 创建目录
            else:
                cached_text_embeddings = {}  # 使用内存缓存

            # 遍历所有文本文件进行预计算
            for txt in tqdm([i for i in os.listdir(args.data_config.img_dir) if ".txt" in i]):
                txt_path = os.path.join(args.data_config.img_dir, txt)  # 文本文件完整路径
                prompt = open(txt_path).read()  # 读取提示词内容

                # 使用文本编码管道编码提示词
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    prompt=[prompt],  # 提示词列表
                    device=text_encoding_pipeline.device,  # 计算设备
                    num_images_per_prompt=1,  # 每个提示词生成的图像数
                    max_sequence_length=1024,  # 最大序列长度
                )

                # 保存编码结果
                if args.save_cache_on_disk:
                    # 保存到磁盘（节省内存但增加I/O）
                    torch.save({
                        'prompt_embeds': prompt_embeds[0].to('cpu'),  # 移到CPU节省显存
                        'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                    }, os.path.join(txt_cache_dir, txt + '.pt'))
                else:
                    # 保存到内存（快速访问但占用内存）
                    cached_text_embeddings[txt] = {
                        'prompt_embeds': prompt_embeds[0].to('cpu'),
                        'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                    }

            # 计算空提示词嵌入（用于无条件生成）
            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                prompt=[' '],  # 空提示词
                device=text_encoding_pipeline.device,
                num_images_per_prompt=1,
                max_sequence_length=1024,
            )

            # 保存空嵌入
            if args.save_cache_on_disk:
                torch.save({
                    'prompt_embeds': prompt_embeds[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                }, os.path.join(txt_cache_dir, 'empty_embedding.pt'))
                del prompt_embeds  # 立即删除以释放内存
                del prompt_embeds_mask
            else:
                cached_text_embeddings['empty_embedding'] = {
                    'prompt_embeds': prompt_embeds[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                }

        # 8. 文本编码管道清理（释放显存）
        text_encoding_pipeline.to("cpu")  # 移动到CPU
        torch.cuda.empty_cache()  # 清空CUDA缓存

    # 完全删除文本编码管道以释放内存
    del text_encoding_pipeline
    gc.collect()  # 强制垃圾回收

    # 9. VAE 初始化（用于图像编码）
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,  # 预训练模型路径
        subfolder="vae",  # VAE 子文件夹
    )
    vae.to(accelerator.device, dtype=weight_dtype)  # 移动到计算设备并设置数据类型

    # 初始化图像缓存变量
    cached_image_embeddings = None  # 内存中的图像嵌入缓存
    img_cache_dir = None  # 磁盘图像嵌入缓存目录

    # 10. 图像嵌入预计算（内存优化关键步骤2）
    if args.precompute_image_embeddings:
        # 选择缓存策略
        if args.save_cache_on_disk:
            img_cache_dir = os.path.join(cache_dir, "img_embs")  # 图像嵌入磁盘缓存目录
            os.makedirs(img_cache_dir, exist_ok=True)  # 创建目录
        else:
            cached_image_embeddings = {}  # 使用内存缓存

        with torch.no_grad():  # 禁用梯度计算
            # 遍历所有图像文件进行预计算
            for img_name in tqdm([i for i in os.listdir(args.data_config.img_dir)
                                if ".png" in i or ".jpg" in i]):
                # 图像加载和预处理
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                img = image_resize(img, args.data_config.img_size)  # 调整图像大小

                # 确保图像尺寸是32的倍数（VAE要求）
                w, h = img.size
                new_w = (w // 32) * 32  # 宽度对齐到32的倍数
                new_h = (h // 32) * 32  # 高度对齐到32的倍数
                img = img.resize((new_w, new_h))  # 重新调整大小

                # 图像数据转换：PIL -> numpy -> tensor
                img = torch.from_numpy((np.array(img) / 127.5) - 1)  # 归一化到[-1, 1]
                img = img.permute(2, 0, 1).unsqueeze(0)  # 调整维度：HWC -> BCHW
                pixel_values = img.unsqueeze(2)  # 添加时间维度：BCHW -> BCTHW
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)

                # 使用VAE编码图像
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]

                # 保存编码结果
                if args.save_cache_on_disk:
                    torch.save(pixel_latents, os.path.join(img_cache_dir, img_name + '.pt'))
                    del pixel_latents  # 立即删除以释放内存
                else:
                    cached_image_embeddings[img_name] = pixel_latents

        # VAE清理（释放显存）
        vae.to('cpu')  # 移动到CPU
        torch.cuda.empty_cache()  # 清空CUDA缓存

    # 强制垃圾回收
    gc.collect()

    # 11. Transformer 模型加载
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,  # 预训练模型路径
        subfolder="transformer",  # transformer 子文件夹
    )

    # 12. 模型量化（内存优化关键步骤3）
    if args.quantize:
        torch_dtype = weight_dtype  # 保存原始数据类型
        device = accelerator.device  # 计算设备

        # 逐块量化策略（避免显存峰值）
        all_blocks = list(flux_transformer.transformer_blocks)  # 获取所有transformer块
        for block in tqdm(all_blocks):
            block.to(device, dtype=torch_dtype)  # 移动到设备
            quantize(block, weights=qfloat8)  # 量化权重到qfloat8
            freeze(block)  # 冻结量化后的权重
            block.to('cpu')  # 移回CPU以释放显存

        # 量化整个transformer
        flux_transformer.to(device, dtype=torch_dtype)
        quantize(flux_transformer, weights=qfloat8)  # 量化权重
        freeze(flux_transformer)  # 冻结权重

        # 注释掉的备选量化方案（更激进的量化）
        # quantize(flux_transformer, weights=qint8, activations=qint8)
        # freeze(flux_transformer)

    # 13. LoRA 配置
    lora_config = LoraConfig(
        r=args.rank,  # LoRA 秩（低秩分解的维度）
        lora_alpha=args.rank,  # LoRA alpha 参数（缩放因子）
        init_lora_weights="gaussian",  # LoRA 权重初始化方式
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 目标模块（注意力层）
    )

    # 将模型移动到计算设备
    flux_transformer.to(accelerator.device)

    # 14. 噪声调度器初始化
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,  # 预训练模型路径
        subfolder="scheduler",  # 调度器子文件夹
    )

    # 根据是否量化选择不同的设备移动策略
    if args.quantize:
        flux_transformer.to(accelerator.device)  # 量化模型只移动设备
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)  # 非量化模型同时设置数据类型

    # 添加LoRA适配器
    flux_transformer.add_adapter(lora_config)

    # 创建噪声调度器副本（用于训练过程中的计算）
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """
        获取给定时间步的sigma值（噪声强度）
        用于Flow Matching训练中的噪声调度

        Args:
            timesteps: 时间步张量
            n_dim: 目标维度数
            dtype: 数据类型

        Returns:
            sigma: 对应的sigma值张量
        """
        # 获取调度器中的sigma值
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        # 找到每个时间步对应的索引
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        # 获取对应的sigma值
        sigma = sigmas[step_indices].flatten()

        # 扩展维度以匹配目标张量
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # 15. 模型训练准备
    flux_transformer.requires_grad_(False)  # 首先禁用所有参数的梯度
    flux_transformer.train()  # 设置为训练模式

    # 16. 优化器配置
    optimizer_cls = torch.optim.AdamW  # 默认优化器类

    # 设置参数的可训练性：只训练LoRA参数
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:  # 非LoRA参数
            param.requires_grad = False  # 禁用梯度
        else:  # LoRA参数
            param.requires_grad = True  # 启用梯度
            print(n)  # 打印可训练参数名称

    # 计算并打印可训练参数数量
    trainable_params = sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad])
    print(f"{trainable_params / 1000000:.2f}M 可训练参数")

    # 获取LoRA层参数
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())

    # 17. LoRA层管理（内存优化关键步骤4）
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))

    # 启用梯度检查点（以时间换空间）
    flux_transformer.enable_gradient_checkpointing()

    # 18. 优化器初始化（支持8位优化器）
    if args.adam8bit:
        # 使用8位Adam优化器（内存优化关键步骤5）
        optimizer = bnb.optim.Adam8bit(
            lora_layers,  # 只优化LoRA层
            lr=args.learning_rate,  # 学习率
            betas=(args.adam_beta1, args.adam_beta2),  # Adam beta参数
        )
    else:
        # 使用标准AdamW优化器
        optimizer = optimizer_cls(
            lora_layers,  # 只优化LoRA层
            lr=args.learning_rate,  # 学习率
            betas=(args.adam_beta1, args.adam_beta2),  # Adam beta参数
            weight_decay=args.adam_weight_decay,  # 权重衰减
            eps=args.adam_epsilon,  # 数值稳定性参数
        )

    # 19. 数据加载器初始化（支持预计算嵌入）
    train_dataloader = loader(
        cached_text_embeddings=cached_text_embeddings,  # 缓存的文本嵌入
        cached_image_embeddings=cached_image_embeddings,  # 缓存的图像嵌入
        txt_cache_dir=txt_cache_dir,  # 文本嵌入磁盘缓存目录
        img_cache_dir=img_cache_dir,  # 图像嵌入磁盘缓存目录
        **args.data_config  # 其他数据配置参数
    )

    # 20. 学习率调度器初始化
    lr_scheduler = get_scheduler(
        args.lr_scheduler,  # 调度器类型
        optimizer=optimizer,  # 优化器
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,  # 预热步数
        num_training_steps=args.max_train_steps * accelerator.num_processes,  # 总训练步数
    )

    # 21. 全局步数初始化
    global_step = 0

    # 创建测试数据集（用于accelerator.prepare的占位符）
    dataset1 = ToyDataset(num_samples=100, input_dim=10)
    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)

    # 22. Accelerator准备（只准备LoRA层而非整个模型）
    lora_layers_model, optimizer, _, lr_scheduler = accelerator.prepare(
        lora_layers_model,  # LoRA层模型（而非整个transformer）
        optimizer,  # 优化器
        dataloader1,  # 占位符数据加载器
        lr_scheduler  # 学习率调度器
    )

    # 23. 训练初始化
    initial_global_step = 0  # 初始全局步数

    # 初始化训练跟踪器（wandb/tensorboard等）
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    # 计算总批次大小
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # 24. 训练信息记录
    logger.info("***** 开始训练 *****")
    logger.info(f"  每设备瞬时批次大小 = {args.train_batch_size}")
    logger.info(f"  总训练批次大小（包含并行、分布式和累积） = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")

    # 创建进度条
    progress_bar = tqdm(
        range(0, args.max_train_steps),  # 总步数范围
        initial=initial_global_step,  # 初始步数
        desc="训练步数",  # 描述
        disable=not accelerator.is_local_main_process,  # 只在主进程显示
    )

    # 计算VAE缩放因子
    vae_scale_factor = 2 ** len(vae.temperal_downsample)

    # 25. 主训练循环
    for epoch in range(1):  # 只训练一个epoch
        train_loss = 0.0  # 训练损失累积

        # 遍历数据批次
        for step, batch in enumerate(train_dataloader):
            # 梯度累积上下文
            with accelerator.accumulate(flux_transformer):
                # 根据是否预计算文本嵌入选择不同的数据解包方式
                if args.precompute_text_embeddings:
                    # 使用预计算的文本嵌入
                    img, prompt_embeds, prompt_embeds_mask = batch
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype).to(accelerator.device)
                    prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32).to(accelerator.device)
                else:
                    # 使用原始文本提示
                    img, prompts = batch

                # 禁用梯度计算的数据预处理
                with torch.no_grad():
                    # 图像处理：根据是否预计算选择不同策略
                    if not args.precompute_image_embeddings:
                        # 实时编码图像
                        pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                        pixel_values = pixel_values.unsqueeze(2)  # 添加时间维度
                        pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    else:
                        # 使用预计算的图像嵌入
                        pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)

                    # 调整维度顺序：BCTHW -> BTCHW
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

                    # VAE潜在空间标准化
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std

                    # 26. 噪声和时间步采样
                    bsz = pixel_latents.shape[0]  # 批次大小
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)

                    # 计算时间步采样密度
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",  # 权重方案
                        batch_size=bsz,  # 批次大小
                        logit_mean=0.0,  # logit均值
                        logit_std=1.0,  # logit标准差
                        mode_scale=1.29,  # 模式缩放
                    )

                    # 生成时间步索引和时间步
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # 27. Flow Matching 噪声调度
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # 28. 潜在表示打包（为transformer准备输入）
                packed_noisy_model_input = QwenImagePipeline._pack_latents(
                    noisy_model_input,  # 噪声输入
                    bsz,  # 批次大小
                    noisy_model_input.shape[2],  # 通道数
                    noisy_model_input.shape[3],  # 高度
                    noisy_model_input.shape[4],  # 宽度
                )

                # 为RoPE（旋转位置编码）准备图像形状信息
                img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz

                # 29. 文本编码处理
                with torch.no_grad():
                    if not args.precompute_text_embeddings:
                        # 实时编码文本（如果没有预计算）
                        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                            prompt=prompts,  # 文本提示
                            device=packed_noisy_model_input.device,  # 计算设备
                            num_images_per_prompt=1,  # 每个提示生成的图像数
                            max_sequence_length=1024,  # 最大序列长度
                        )

                    # 计算文本序列长度
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                # 30. Transformer 前向传播
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,  # 打包的噪声输入
                    timestep=timesteps / 1000,  # 归一化的时间步
                    guidance=None,  # 引导信号（此处为None）
                    encoder_hidden_states_mask=prompt_embeds_mask,  # 文本编码掩码
                    encoder_hidden_states=prompt_embeds,  # 文本编码
                    img_shapes=img_shapes,  # 图像形状信息
                    txt_seq_lens=txt_seq_lens,  # 文本序列长度
                    return_dict=False,  # 不返回字典格式
                )[0]  # 取第一个输出

                # 31. 潜在表示解包
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,  # 模型预测
                    height=noisy_model_input.shape[3] * vae_scale_factor,  # 原始高度
                    width=noisy_model_input.shape[4] * vae_scale_factor,  # 原始宽度
                    vae_scale_factor=vae_scale_factor,  # VAE缩放因子
                )

                # 32. 损失计算
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

                # Flow Matching 损失目标
                target = noise - pixel_latents  # 目标是噪声与原始潜在表示的差
                target = target.permute(0, 2, 1, 3, 4)  # 调整维度顺序

                # 计算均方误差损失
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,  # 在第1维度上求均值
                )
                loss = loss.mean()  # 最终损失

                # 33. 分布式训练损失聚合
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 34. 反向传播
                accelerator.backward(loss)  # 反向传播

                # 梯度裁剪（如果启用了梯度同步）
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)

                # 优化器步骤
                optimizer.step()  # 更新参数
                lr_scheduler.step()  # 更新学习率
                optimizer.zero_grad()  # 清零梯度

            # 35. 训练步骤完成后的处理
            if accelerator.sync_gradients:
                progress_bar.update(1)  # 更新进度条
                global_step += 1  # 增加全局步数
                accelerator.log({"train_loss": train_loss}, step=global_step)  # 记录损失
                train_loss = 0.0  # 重置训练损失

                # 36. 检查点保存
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:  # 只在主进程保存
                        # 检查是否需要删除旧检查点
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # 删除超出限制的旧检查点
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} 个检查点已存在，删除 {len(removing_checkpoints)} 个检查点"
                                )
                                logger.info(f"删除检查点: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)  # 删除目录

                    # 保存当前检查点
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    # 创建保存目录
                    try:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                    except:
                        pass  # 忽略创建目录的错误

                    # 提取并保存LoRA权重
                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )

                    # 使用安全序列化保存LoRA权重
                    QwenImagePipeline.save_lora_weights(
                        save_path,  # 保存路径
                        flux_transformer_lora_state_dict,  # LoRA状态字典
                        safe_serialization=True,  # 使用安全序列化
                    )

                    logger.info(f"检查点已保存到 {save_path}")

            # 37. 更新训练日志
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)  # 在进度条中显示

            # 检查是否达到最大训练步数
            if global_step >= args.max_train_steps:
                break

    # 38. 训练结束
    accelerator.wait_for_everyone()  # 等待所有进程完成
    accelerator.end_training()  # 结束训练


if __name__ == "__main__":
    main()  # 运行主函数
