# 导入命令行参数解析库
import argparse
# 导入深拷贝功能
import copy
from copy import deepcopy
# 导入日志记录库
import logging
# 导入操作系统接口库
import os
# 导入文件操作库
import shutil

# 导入PyTorch深度学习框架
import torch
# 导入进度条显示库
from tqdm.auto import tqdm

# 导入Accelerate分布式训练加速库
from accelerate import Accelerator
# 导入Accelerate日志记录器
from accelerate.logging import get_logger
# 导入Accelerate项目配置
from accelerate.utils import ProjectConfiguration
# 导入数据集处理库
import datasets
# 导入Diffusers扩散模型库
import diffusers
# 导入Flow Match Euler离散调度器
from diffusers import FlowMatchEulerDiscreteScheduler
# 导入Qwen图像相关模型组件
from diffusers import (
    AutoencoderKLQwenImage,  # Qwen图像自编码器
    QwenImagePipeline,       # Qwen图像生成管道
    QwenImageTransformer2DModel,  # Qwen图像Transformer模型
)
# 导入优化器调度器
from diffusers.optimization import get_scheduler
# 导入训练工具函数
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,  # 计算时间步采样密度
    compute_loss_weighting_for_sd3,         # 计算SD3损失权重
)
# 导入状态字典转换工具
from diffusers.utils import convert_state_dict_to_diffusers
# 导入PyTorch工具函数
from diffusers.utils.torch_utils import is_compiled_module
# 导入图像数据集加载器
from image_datasets.dataset import loader
# 导入配置文件解析库
from omegaconf import OmegaConf
# 导入LoRA配置
from peft import LoraConfig
# 导入PEFT模型状态字典获取工具
from peft.utils import get_peft_model_state_dict
# 导入Transformers库
import transformers

# 创建日志记录器，设置日志级别为INFO
logger = get_logger(__name__, log_level="INFO")




def parse_args():
    """解析命令行参数的函数"""
    # 创建参数解析器，添加描述信息
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # 添加配置文件路径参数
    parser.add_argument(
        "--config",          # 参数名
        type=str,           # 参数类型为字符串
        default=None,       # 默认值为None
        required=True,      # 必需参数
        help="path to config",  # 帮助信息
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 返回配置文件路径
    return args.config


def main():
    """主训练函数"""
    # 使用OmegaConf加载配置文件
    args = OmegaConf.load(parse_args())
    # 构建日志目录路径
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # 创建Accelerator项目配置
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # 初始化Accelerator分布式训练加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数
        mixed_precision=args.mixed_precision,                          # 混合精度训练
        log_with=args.report_to,                                      # 日志记录方式
        project_config=accelerator_project_config,                    # 项目配置
    )
    
    def unwrap_model(model):
        """解包模型的辅助函数，用于获取原始模型"""
        # 使用accelerator解包模型
        model = accelerator.unwrap_model(model)
        # 如果是编译模块，获取原始模块
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 配置日志记录格式和级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # 日志格式
        datefmt="%m/%d/%Y %H:%M:%S",                                    # 日期格式
        level=logging.INFO,                                             # 日志级别
    )
    # 记录accelerator状态信息
    logger.info(accelerator.state, main_process_only=False)
    
    # 根据是否为本地主进程设置不同的日志级别
    if accelerator.is_local_main_process:
        # 主进程设置警告级别
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # 非主进程设置错误级别
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    # 如果是主进程，创建输出目录
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置权重数据类型，默认为float32
    weight_dtype = torch.float32
    # 根据混合精度设置调整权重数据类型
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    # 加载文本编码管道，不包含transformer和vae
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    
    # 加载VAE（变分自编码器）
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",  # 从vae子文件夹加载
    )
    
    # 加载Flux Transformer模型
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",  # 从transformer子文件夹加载
    )
    
    # 配置LoRA（Low-Rank Adaptation）参数
    lora_config = LoraConfig(
        r=args.rank,                                              # LoRA秩
        lora_alpha=args.rank,                                     # LoRA alpha参数
        init_lora_weights="gaussian",                             # 初始化方式为高斯分布
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],    # 目标模块
    )
    
    # 加载噪声调度器
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",  # 从scheduler子文件夹加载
    )
    
    # 将模型移动到指定设备并设置数据类型
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    # 为transformer添加LoRA适配器
    flux_transformer.add_adapter(lora_config)
    # 将文本编码管道移动到指定设备
    text_encoding_pipeline.to(accelerator.device)
    # 深拷贝噪声调度器
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """获取sigma值的函数，用于噪声调度"""
        # 将sigma值移动到指定设备和数据类型
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        # 获取调度时间步
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        # 将输入时间步移动到指定设备
        timesteps = timesteps.to(accelerator.device)
        # 找到每个时间步对应的索引
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    
        # 根据索引获取对应的sigma值
        sigma = sigmas[step_indices].flatten()
        # 扩展维度以匹配所需的维度数
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    # 冻结VAE参数，不参与训练
    vae.requires_grad_(False)
    # 冻结transformer参数，稍后只训练LoRA部分
    flux_transformer.requires_grad_(False)


    # 设置transformer为训练模式
    flux_transformer.train()
    # 设置优化器类为AdamW
    optimizer_cls = torch.optim.AdamW
    
    # 遍历transformer的所有参数
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            # 非LoRA参数不参与梯度计算
            param.requires_grad = False
            pass
        else:
            # LoRA参数参与梯度计算
            param.requires_grad = True
            print(n)  # 打印LoRA参数名称
    
    # 计算并打印可训练参数数量（以百万为单位）
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    # 筛选出需要梯度的LoRA层参数
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())

    # 启用梯度检查点以节省显存
    flux_transformer.enable_gradient_checkpointing()
    
    # 初始化优化器
    optimizer = optimizer_cls(
        lora_layers,                        # 优化的参数
        lr=args.learning_rate,              # 学习率
        betas=(args.adam_beta1, args.adam_beta2),  # Adam的beta参数
        weight_decay=args.adam_weight_decay,       # 权重衰减
        eps=args.adam_epsilon,                     # epsilon参数
    )

    # 创建训练数据加载器
    train_dataloader = loader(**args.data_config)    

    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,                                              # 调度器类型
        optimizer=optimizer,                                            # 优化器
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,     # 预热步数
        num_training_steps=args.max_train_steps * accelerator.num_processes,   # 总训练步数
    )
    
    # 初始化全局步数
    global_step = 0
    # 将VAE移动到指定设备和数据类型
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # 使用accelerator准备模型、优化器和调度器
    flux_transformer, optimizer, _, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, deepcopy(train_dataloader), lr_scheduler
    )


    # 设置初始全局步数
    initial_global_step = 0

    # 如果是主进程，初始化训练跟踪器
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    # 计算总批次大小
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # 记录训练信息
    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    # 创建进度条
    progress_bar = tqdm(
        range(0, args.max_train_steps),                    # 总步数范围
        initial=initial_global_step,                       # 初始步数
        desc="Steps",                                      # 描述
        disable=not accelerator.is_local_main_process,    # 是否禁用
    )
    
    # 计算VAE缩放因子
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    
    # 开始训练循环（只训练1个epoch）
    for epoch in range(1):
        train_loss = 0.0  # 初始化训练损失
        
        # 遍历训练数据
        for step, batch in enumerate(train_dataloader):
            # 使用accelerator的梯度累积上下文
            with accelerator.accumulate(flux_transformer):
                # 解包批次数据：图像和提示词
                img, prompts = batch
                
                # 在不计算梯度的上下文中处理数据
                with torch.no_grad():
                    # 将图像转换为指定数据类型并移动到设备
                    pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                    # 在第2维增加一个维度
                    pixel_values = pixel_values.unsqueeze(2)

                    # 使用VAE编码图像得到潜在表示
                    pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    # 调整维度顺序
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

                    # 获取潜在空间的均值
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    # 计算潜在空间的标准差倒数
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    # 标准化潜在表示
                    pixel_latents = (pixel_latents - latents_mean) * latents_std
                    

                    # 获取批次大小
                    bsz = pixel_latents.shape[0]
                    # 生成与潜在表示相同形状的随机噪声
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    
                    # 计算时间步采样的密度
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",  # 权重方案
                        batch_size=bsz,          # 批次大小
                        logit_mean=0.0,          # logit均值
                        logit_std=1.0,           # logit标准差
                        mode_scale=1.29,         # 模式缩放
                    )
                    # 计算时间步索引
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    # 获取对应的时间步
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # 获取sigma值
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                # 计算加噪后的模型输入
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                
                # 打包潜在表示
                packed_noisy_model_input = QwenImagePipeline._pack_latents(
                    noisy_model_input,                    # 加噪输入
                    bsz,                                  # 批次大小
                    noisy_model_input.shape[2],          # 通道数
                    noisy_model_input.shape[3],          # 高度
                    noisy_model_input.shape[4],          # 宽度
                )
                
                # 为RoPE准备图像形状信息
                img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz
                
                # 在不计算梯度的上下文中编码提示词
                with torch.no_grad():
                    # 使用文本编码管道编码提示词
                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                        prompt=prompts,                           # 提示词
                        device=packed_noisy_model_input.device,  # 设备
                        num_images_per_prompt=1,                 # 每个提示词的图像数
                        max_sequence_length=1024,               # 最大序列长度
                    )
                    # 计算文本序列长度
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                
                # 使用transformer模型进行前向传播
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,      # 隐藏状态
                    timestep=timesteps / 1000,                   # 时间步（归一化）
                    guidance=None,                               # 引导信息
                    encoder_hidden_states_mask=prompt_embeds_mask,  # 编码器隐藏状态掩码
                    encoder_hidden_states=prompt_embeds,         # 编码器隐藏状态
                    img_shapes=img_shapes,                       # 图像形状
                    txt_seq_lens=txt_seq_lens,                  # 文本序列长度
                    return_dict=False,                          # 不返回字典
                )[0]
                
                # 解包模型预测结果
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,                                          # 模型预测
                    height=noisy_model_input.shape[3] * vae_scale_factor,  # 高度
                    width=noisy_model_input.shape[4] * vae_scale_factor,   # 宽度
                    vae_scale_factor=vae_scale_factor,                     # VAE缩放因子
                )
                
                # 计算损失权重
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                
                # 计算flow-matching损失的目标
                target = noise - pixel_latents
                # 调整目标的维度顺序
                target = target.permute(0, 2, 1, 3, 4)
                
                # 计算损失
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # 收集所有进程的损失用于日志记录
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                
                # 如果需要同步梯度
                if accelerator.sync_gradients:
                    # 梯度裁剪
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                
                # 优化器步进
                optimizer.step()
                # 学习率调度器步进
                lr_scheduler.step()
                # 清零梯度
                optimizer.zero_grad()

            # 检查accelerator是否执行了优化步骤
            if accelerator.sync_gradients:
                # 更新进度条
                progress_bar.update(1)
                global_step += 1
                # 记录训练损失
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # 检查是否需要保存检查点
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # 检查检查点总数限制
                        if args.checkpoints_total_limit is not None:
                            # 获取现有检查点列表
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # 如果检查点数量超过限制，删除旧的检查点
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                # 删除旧检查点
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    # 构建保存路径
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    # 创建保存目录
                    try:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                    except:
                        pass
                    
                    # 解包模型
                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    # 获取LoRA状态字典
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )

                    # 保存LoRA权重
                    QwenImagePipeline.save_lora_weights(
                        save_path,                              # 保存路径
                        flux_transformer_lora_state_dict,      # LoRA状态字典
                        safe_serialization=True,               # 安全序列化
                    )

                    logger.info(f"Saved state to {save_path}")

            # 记录步骤损失和学习率
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # 检查是否达到最大训练步数
            if global_step >= args.max_train_steps:
                break

    # 等待所有进程完成
    accelerator.wait_for_everyone()
    # 结束训练
    accelerator.end_training()


# 程序入口点
if __name__ == "__main__":
    main()
