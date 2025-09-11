"""
简单示例：从检查点加载训练好的Qwen图像模型
这是一个加载训练好的模型检查点的最小示例
"""

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
import torch
from omegaconf import OmegaConf
import os


def load_trained_model(checkpoint_path):
    """从检查点加载训练好的模型 - 简化版本"""
    print(f"正在从以下路径加载训练好的模型: {checkpoint_path}")
    
    # 加载配置文件以获取原始模型路径
    config_path = os.path.join(checkpoint_path, "config.yaml")
    config = OmegaConf.load(config_path)
    original_model_path = config.pretrained_model_name_or_path
    
    # 加载训练好的transformer
    transformer_path = os.path.join(checkpoint_path, "transformer")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    transformer.to("cuda")
    transformer.eval()
    
    # 从原始模型加载VAE
    vae = AutoencoderKLQwenImage.from_pretrained(
        original_model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )
    vae.to("cuda")
    vae.eval()
    
    # 创建管道
    pipe = QwenImagePipeline.from_pretrained(
        original_model_path,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    print("模型加载成功!")
    return pipe


def main():
    """加载和使用训练好的模型的简单示例"""
    # 1. 加载训练好的模型
    checkpoint_path = "output_full_training/checkpoint-500"
    pipe = load_trained_model(checkpoint_path)
    
    # 2. 生成图像
    prompt = "一个美丽的风景，有山脉和湖泊"
    
    image = pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=40,
        true_cfg_scale=5,
        generator=torch.Generator(device="cuda").manual_seed(42)
    )
    
    # 3. 保存结果
    output_image = image.images[0]
    output_image.save("generated_image.png")
    print("图像已保存为: generated_image.png")


if __name__ == "__main__":
    main()
