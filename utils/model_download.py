"""
模型下载脚本
"""
from modelscope import snapshot_download
import os

def download_model(model_id="Qwen/Qwen-Image", local_dir='./models/Qwen-Image'):
    """
    下载模型
    
    Args:
        model_id (str): 模型ID
        local_dir (str): 模型本地保存路径
    """
    print(f"开始下载 {model_id} 模型到: {local_dir}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    
    # 下载模型
    snapshot_download(
        model_id, 
        local_dir=local_dir
    )
    
    print(f"模型下载完成，保存在: {local_dir}")

if __name__ == "__main__":
    # 默认下载到当前目录的 models 文件夹
    download_model()
