# Qwen-Image & Qwen-Image-Edit 的 LoRA 训练

由 [FlyMy.AI](https://flymy.ai) 提供的开源实现，用于训练 Qwen/Qwen-Image 和 Qwen/Qwen-Image-Edit 模型的 LoRA（低秩适应）层。

<p align="center">
  <img src="./assets/flymy_transparent.png" alt="FlyMy.AI Logo" width="256">
</p>

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=FlyMyAI/flymyai-lora-trainer&type=Date)](https://www.star-history.com/#FlyMyAI/flymyai-lora-trainer&Date)

## 🌟 关于 FlyMy.AI

GenAI 的智能体基础设施。FlyMy.AI 是一个用于构建和运行 GenAI 媒体智能体的 B2B 基础设施。

**🔗 有用链接：**
- 🌐 [官方网站](https://flymy.ai)
- 📚 [文档](https://docs.flymy.ai/intro)
- 💬 [Discord 社区](https://discord.com/invite/t6hPBpSebw)
- 🤗 [预训练 LoRA 模型](https://huggingface.co/flymy-ai/qwen-image-realism-lora)
- 🐦 [X (Twitter)](https://x.com/flymyai)
- 💼 [LinkedIn](https://linkedin.com/company/flymyai)
- 📺 [YouTube](https://youtube.com/@flymyai)
- 📸 [Instagram](https://www.instagram.com/flymy_ai)

## 🚀 特性

- 基于 LoRA 的高效微调训练
- 支持 Qwen-Image 和 Qwen-Image-Edit 模型
- 兼容 Hugging Face `diffusers`
- 通过 YAML 轻松配置
- 基于控制的 LoRA 图像编辑
- LoRA 训练的开源实现
- Qwen-Image 的完整训练

## 📅 更新

**02.09.2025**
- ✅ 添加了 Qwen-Image 和 Qwen-Image-Edit 的完整训练

**20.08.2025**
- ✅ 添加了 Qwen-Image-Edit LoRA 训练器支持

**09.08.2025**
- ✅ 添加了 < 24GiB GPU 的训练管道

**08.08.2025**
- ✅ 添加了全面的数据集准备说明
- ✅ 添加了数据集验证脚本（`utils/validate_dataset.py`）
- ✅ 训练期间冻结模型权重

## ⚠️ 项目状态

**🚧 开发中：** 我们正在积极改进代码并添加测试覆盖率。项目处于完善阶段但已可使用。

**📋 开发计划：**
- ✅ 基础代码正常工作
- ✅ 训练功能已实现
- 🔄 性能优化进行中
- 🔜 测试覆盖率即将推出

---

## 📦 安装

**要求：**
- Python 3.10

1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/FlyMyAI/flymyai-lora-trainer
   cd flymyai-lora-trainer
   ```

2. 安装所需包：
   ```bash
   pip install -r requirements.txt
   ```

3. 从 GitHub 安装最新的 `diffusers`：
   ```bash
   pip install git+https://github.com/huggingface/diffusers
   ```

4. 下载预训练的 LoRA 权重（可选）：
   ```bash
   # 克隆包含 LoRA 权重的仓库
   git clone https://huggingface.co/flymy-ai/qwen-image-realism-lora
   
   # 或下载特定文件
   wget https://huggingface.co/flymy-ai/qwen-image-realism-lora/resolve/main/flymy_realism.safetensors
   ```

---

## 📁 数据准备

### Qwen-Image 训练的数据集结构

训练数据应遵循与 Flux LoRA 训练相同的格式，每个图像都有一个同名的对应文本文件：

```
dataset/
├── img1.png
├── img1.txt
├── img2.jpg
├── img2.txt
├── img3.png
├── img3.txt
└── ...
```

### Qwen-Image-Edit 训练的数据集结构

对于基于控制的图像编辑，数据集应该组织为目标图像/标题和控制图像的单独目录：

```
dataset/
├── images/           # 目标图像及其标题
│   ├── image_001.jpg
│   ├── image_001.txt
│   ├── image_002.jpg
│   ├── image_002.txt
│   └── ...
└── control/          # 控制图像
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

### 数据格式要求

1. **图像**：支持常见格式（PNG、JPG、JPEG、WEBP）
2. **文本文件**：包含图像描述的纯文本文件
3. **文件命名**：每个图像必须有一个同名的对应文本文件

### 示例数据结构

```
my_training_data/
├── portrait_001.png
├── portrait_001.txt
├── landscape_042.jpg
├── landscape_042.txt
├── abstract_design.png
├── abstract_design.txt
└── style_reference.jpg
└── style_reference.txt
```

### 文本文件内容示例

**portrait_001.txt:**
```
一位棕色头发年轻女性的真实肖像，自然光照，专业摄影风格
```

**landscape_042.txt:**
```
日落时分的山景，戏剧性云彩，黄金时刻光照，广角视图
```

**abstract_design.txt:**
```
具有几何形状的现代抽象艺术，鲜艳色彩，极简主义构图
```

### 数据准备技巧

1. **图像质量**：使用高分辨率图像（推荐 1024x1024 或更高）
2. **描述质量**：为您的图像编写详细、准确的描述
3. **一致性**：在数据集中保持一致的风格和质量
4. **数据集大小**：为了获得良好结果，至少使用 10-50 个图像-文本对
5. **触发词**：如果训练特定概念，在描述中包含一致的触发词
6. **自动生成描述**：您可以使用 [Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2) 自动生成图像描述

### 快速数据验证

您可以使用包含的验证工具验证数据结构：

```bash
python utils/validate_dataset.py --path path/to/your/dataset
```

这将检查：
- 每个图像都有对应的文本文件
- 所有文件都遵循正确的命名约定
- 报告任何缺失文件或不一致之处

---

## 🏁 在 < 24gb 显存上开始训练

要使用您的配置文件（例如 `train_lora_4090.yaml`）开始训练，运行：

```bash
accelerate launch train_4090.py --config ./train_configs/train_lora_4090.yaml
```
![示例输出](./assets/Valentin_24gb.jpg)

## 🏁 开始训练

### Qwen-Image LoRA 训练

要使用您的配置文件（例如 `train_lora.yaml`）开始训练，运行：

```bash
accelerate launch train.py --config ./train_configs/train_lora.yaml
```

确保 `train_lora.yaml` 正确设置了数据集、模型、输出目录和其他参数的路径。

### Qwen-Image 完整训练

要使用您的配置文件（例如 `train_full_qwen_image.yaml`）开始训练，运行：

```bash
accelerate launch train_full_qwen_image.py --config ./train_configs/train_full_qwen_image.yaml
```

确保 `train_full_qwen_image.yaml` 正确设置了数据集、模型、输出目录和其他参数的路径。

#### 加载训练的完整模型

训练后，您可以从检查点目录加载训练的模型进行推理。

**简单示例：**

```python
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
import torch
from omegaconf import OmegaConf
import os

def load_trained_model(checkpoint_path):
    """从检查点加载训练的模型"""
    print(f"从以下位置加载训练的模型：{checkpoint_path}")
    
    # 加载配置以获取原始模型路径
    config_path = os.path.join(checkpoint_path, "config.yaml")
    config = OmegaConf.load(config_path)
    original_model_path = config.pretrained_model_name_or_path
    
    # 加载训练的 transformer
    transformer_path = os.path.join(checkpoint_path, "transformer")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    transformer.to("cuda")
    transformer.eval()
    
    # 从原始模型加载 VAE
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
    
    print("模型加载成功！")
    return pipe

# 使用方法
checkpoint_path = "/path/to/your/checkpoint"
pipe = load_trained_model(checkpoint_path)

# 生成图像
prompt = "美丽的山湖风景"
image = pipe(
    prompt=prompt,
    width=768,
    height=768,
    num_inference_steps=30,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(42)
)

# 保存结果
output_image = image.images[0]
output_image.save("generated_image.png")
```

**完整示例脚本：**

```bash
python inference_trained_model_gpu_optimized.py
```

**检查点结构：**

训练的模型以以下结构保存：
```
checkpoint/
├── config.yaml          # 训练配置
└── transformer/         # 训练的 transformer 权重
    ├── config.json
    ├── diffusion_pytorch_model.safetensors.index.json
    └── diffusion_pytorch_model-00001-of-00005.safetensors
    └── ... (多个分片文件)
```

### Qwen-Image-Edit LoRA 训练

对于基于控制的图像编辑训练，使用专门的训练脚本：

```bash
accelerate launch train_qwen_edit_lora.py --config ./train_configs/train_lora_qwen_edit.yaml
```

#### Qwen-Image-Edit 的配置

配置文件 `train_lora_qwen_edit.yaml` 应包括：

- `img_dir`：目标图像和标题目录的路径（例如 `./extracted_dataset/train/images`）
- `control_dir`：控制图像目录的路径（例如 `./extracted_dataset/train/control`）
- 其他标准 LoRA 训练参数

## 🧪 使用

### Qwen-Image-Edit 完整训练

要使用您的配置文件（例如 `train_full_qwen_edit.yaml`）开始训练，运行：

```bash
accelerate launch train_full_qwen_edit.py --config ./train_configs/train_full_qwen_edit.yaml
```

---

### 🔧 Qwen-Image 初始化

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# 加载管道
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)
```

### 🔧 Qwen-Image-Edit 初始化

```python
from diffusers import QwenImageEditPipeline
import torch
from PIL import Image

# 加载管道
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
```

### 🔌 加载 LoRA 权重

对于 Qwen-Image：
```python
# 加载 LoRA 权重
pipe.load_lora_weights('flymy-ai/qwen-image-realism-lora', adapter_name="lora")
```

对于 Qwen-Image-Edit：
```python
# 加载训练的 LoRA 权重
pipeline.load_lora_weights("/path/to/your/trained/lora/pytorch_lora_weights.safetensors")
```

### 🎨 使用 Qwen-Image LoRA 生成图像
您可以在[这里](https://huggingface.co/flymy-ai/qwen-image-realism-lora)找到 LoRA 权重

无需触发词
```python
prompt = '''非洲裔青少年女性的超现实主义肖像，宁静平和，双臂交叉，戏剧性工作室灯光照明，阳光公园背景，佩戴精致珠宝，四分之三视角，阳光亲吻的肌肤带有自然瑕疵，松散的齐肩卷发，微眯的眼睛，环境街头肖像，T恤上有"FLYMY AI"文字。'''
negative_prompt = " "
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(346346)
)

# 显示图像（在 Jupyter 中或保存到文件）
image.show()
# 或
image.save("output.png")
```

### 🎨 使用 Qwen-Image-Edit LoRA 编辑图像

```python
# 加载输入图像
image = Image.open("/path/to/your/input/image.jpg").convert("RGB")

# 定义编辑提示
prompt = "在同一场景中拍摄人物远离相机的镜头，保持相机稳定以保持对中心主体的焦点，逐渐缩小以捕捉更多周围环境，随着人物在远处变得不那么详细。"

# 生成编辑的图像
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("edited_image.png")
```

### 🖼️ 示例输出 - Qwen-Image

![示例输出](./assets/lora.png)

### 🖼️ 示例输出 - Qwen-Image-Edit

**输入图像：**

![输入图像](./assets/qie2_orig.jpg)

**提示：** 
"在同一场景中拍摄左手固定切菜板边缘而右手倾斜它的镜头，使切碎的番茄滑入锅中，相机角度稍微向左移动以更多地聚焦在锅上。"

**不使用 LoRA 的输出：**

![不使用 LoRA 的输出](./assets/qie2_orig.jpg)

**使用 LoRA 的输出：**

![使用 LoRA 的输出](./assets/qie2_lora.jpg)

---

## 🎛️ 与 ComfyUI 一起使用

我们提供了一个即用型的 ComfyUI 工作流，可与我们训练的 LoRA 模型配合使用。按照以下步骤设置和使用工作流：

### 设置说明

1. **下载最新的 ComfyUI**：
   - 访问 [ComfyUI GitHub 仓库](https://github.com/comfyanonymous/ComfyUI)
   - 克隆或下载最新版本

2. **安装 ComfyUI**：
   - 按照 [ComfyUI 仓库](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#installing) 的安装说明
   - 确保所有依赖项都正确安装

3. **下载 Qwen-Image 模型权重**：
   - 前往 [Qwen-Image ComfyUI 权重](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main)
   - 下载所有模型文件

4. **将 Qwen-Image 权重放入 ComfyUI**：
   - 将下载的 Qwen-Image 模型文件复制到 `ComfyUI/models/` 中的相应文件夹
   - 按照模型仓库中指定的文件夹结构

5. **下载我们的预训练 LoRA 权重**：
   - 访问 [flymy-ai/qwen-image-lora](https://huggingface.co/flymy-ai/qwen-image-lora)
   - 下载 LoRA `.safetensors` 文件

6. **将 LoRA 权重放入 ComfyUI**：
   - 将 LoRA 文件 `flymy-ai/qwen-image-lora/pytorch_lora_weights.safetensors` 复制到 `ComfyUI/models/loras/`

7. **加载工作流**：
   - 在浏览器中打开 ComfyUI
   - 加载位于此仓库中的工作流文件 `qwen_image_lora_example.json`
   - 工作流已预配置为与我们的 LoRA 模型配合使用

### 工作流特性

- ✅ 为 Qwen-Image + LoRA 推理预配置
- ✅ 优化设置以获得最佳质量输出
- ✅ 轻松调整提示和参数
- ✅ 兼容我们所有训练的 LoRA 模型

ComfyUI 工作流提供了一个用户友好的界面，用于使用我们训练的 LoRA 模型生成图像，无需编写 Python 代码。

### 🖼️ 工作流截图

![ComfyUI 工作流](./assets/comfyui_workflow.png)

---

## 🤝 支持

如果您有问题或建议，请加入我们的社区：
- 🌐 [FlyMy.AI](https://flymy.ai)
- 💬 [Discord 社区](https://discord.com/invite/t6hPBpSebw)
- 🐦 [在 X 上关注我们](https://x.com/flymyai)
- 💼 [在 LinkedIn 上联系](https://linkedin.com/company/flymyai)
- 📧 [支持](mailto:support@flymy.ai)

**⭐ 如果您喜欢这个仓库，别忘了给它点星！**
