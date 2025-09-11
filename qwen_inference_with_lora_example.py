"""
简单的Qwen-Image LoRA推理脚本
加载你训练的LoRA权重并生成图片
"""

import torch
from diffusers import QwenImagePipeline


def main():
    # 设置参数
    model_name = "./models/Qwen-Image"
    lora_path = "./output/checkpoint-500/pytorch_lora_weights.safetensors"  # 你的LoRA权重路径
    prompt = "A legendary warrior stands atop a mountain peak, wielding an ancient magical sword that glows with ethereal blue light. Clad in ornate armor with intricate engravings, the figure commands respect and power. The warrior's cape flows dramatically in the wind as storm clouds gather overhead, lightning illuminating the battlefield below. This is a champion of justice, ready to face any challenge with unwavering determination and mystical abilities."
    output_path = "generated_with_lora.png"
    
    # 检查设备
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print("使用CPU")
    
    # 加载基础模型
    print(f"正在加载基础模型: {model_name}")
    pipe = QwenImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)
    
    # 加载LoRA权重
    print(f"正在加载LoRA权重: {lora_path}")
    try:
        # 对于本地文件，需要指定目录而不是具体文件
        import os
        if os.path.isfile(lora_path):
            # 如果是文件路径，取目录
            lora_dir = os.path.dirname(lora_path)
        else:
            # 如果是目录路径，直接使用
            lora_dir = lora_path

        pipe.load_lora_weights(lora_dir)
        print("✅ LoRA权重加载成功")
    except Exception as e:
        print(f"❌ LoRA权重加载失败: {e}")
        return
    
    # 生成图像
    print("正在生成图像...")
    print(f"提示词: {prompt}")
    
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=30,
            true_cfg_scale=5.0,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]
    
    # 保存图像
    image.save(output_path)
    print(f"✅ 图像已保存到: {output_path}")
    print(f"图像尺寸: {image.size}")

if __name__ == "__main__":
    main()
