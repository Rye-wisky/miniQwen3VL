import os
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoConfig

def save_vision_component(model_path="./Qwen3VL-2B", output_dir="./qwen3_vision_standalone"):
    """
    提取 Qwen3-VL 的视觉编码器及其权重，并保存为独立文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在从 {model_path} 加载完整模型以提取组件...")
    
    # 1. 加载完整模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 2. 获取视觉编码器实例
    vision_model = model.model.visual
    
    # 3. 提取并保存 State Dict
    vision_state_dict = vision_model.state_dict()
    weights_path = os.path.join(output_dir, "vision_model_weights.bin")
    torch.save(vision_state_dict, weights_path)
    
    # 4. 保存视觉配置 (这是实例化类所必须的)
    # 我们直接保存 vision_config 对象
    config_path = os.path.join(output_dir, "vision_config.pt")
    torch.save(model.config.vision_config, config_path)

    print(f"成功！权重已保存至: {weights_path}")
    print(f"配置已保存至: {config_path}")
    
    return weights_path, config_path

def load_vision_standalone(output_dir="./qwen3_vision_standalone"):
    """
    直接使用类定义和权重文件实例化视觉编码器，不加载语言模型。
    """
    weights_path = os.path.join(output_dir, "vision_model_weights.bin")
    config_path = os.path.join(output_dir, "vision_config.pt")

    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        raise FileNotFoundError("找不到权重或配置文件，请先运行 save_vision_component")

    print("正在恢复视觉编码器...")

    # 修复 PyTorch 2.6+ 的安全性错误：
    # 因为 config 是自定义类对象，必须设置 weights_only=False
    try:
        vision_config = torch.load(config_path, weights_only=False)
    except TypeError:
        # 兼容旧版本 PyTorch
        vision_config = torch.load(config_path)

    # 获取类定义
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    # 实例化模型
    vision_encoder = Qwen3VLVisionModel(vision_config)

    # 加载权重
    # 权重文件全是 Tensor，通常可以用 weights_only=True，但为了保险也统一设置
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
        
    vision_encoder.load_state_dict(state_dict)
    
    vision_encoder.eval() 
    print("视觉编码器已成功独立加载！")
    return vision_encoder

if __name__ == "__main__":
    # 流程示例
    # 1. 提取 (确保权重已保存)
    # save_vision_component()
    
    # 2. 加载
    standalone_v_model = load_vision_standalone()
    
    print("\n--- 错误修复说明 ---")
    print("PyTorch 2.6+ 默认不允许加载自定义对象。")
    print("已在 torch.load 中添加 weights_only=False 以允许加载 Qwen3VLVisionConfig 类实例。")