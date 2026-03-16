import os
import sys
import time
import json
import argparse
import warnings
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, TextStreamer

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model_minimind_Qwen3VL import MiniQwen3VLConfig, MiniQwen3VLForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')

def init_vlm_model(args):
    print(f"Loading tokenizer from {args.tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    
    print(f"Loading processor from {args.processor_name}")
    processor = AutoProcessor.from_pretrained(args.processor_name, trust_remote_code=True)
    
    # [关键同步] 注入与训练阶段完全一致的特殊 ID
    processor.tokenizer = tokenizer
    processor.image_token_id = 6402  # 对应自定义词表中的 <|image_pad|>
    processor.vision_start_token_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    processor.vision_end_token_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    processor.video_token_id = None  # 禁用视频
    
    # 读取配置
    print(f"Loading configurations...")
    with open(args.text_config_path, 'r', encoding='utf-8') as f:
        text_config_dict = json.load(f)
    vision_config = torch.load(args.vision_config_path, weights_only=False)
    
    config = MiniQwen3VLConfig(
        vision_config=vision_config, 
        text_config=text_config_dict, 
        image_token_id=processor.image_token_id
    )
    
    # 实例化模型
    print("Initializing MiniQwen3VL model...")
    model = MiniQwen3VLForCausalLM(config)
    
    # 加载统一的 Stage2 Checkpoint
    ckp_path = f"{args.save_dir}/{args.weight}.pth"
    print(f"Loading checkpoint from {ckp_path}")
    ckp_data = torch.load(ckp_path, map_location='cpu')
    model.load_state_dict(ckp_data['model'] if 'model' in ckp_data else ckp_data, strict=False)
    
    # 设置精度并移至设备
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = model.to(dtype).to(args.device).eval()
    
    return model, processor

def generate_3d_positions_ids(input_ids, image_grid_thw, processor):
    """
    完全复刻 lm_dataset.py 中的 3D MRoPE 坐标生成逻辑，用于推理时的输入构建。
    """
    seq_len = input_ids.shape[0]
    positions_ids = torch.zeros(3, seq_len, dtype=torch.long)
    step = 0 
    idx = 0  
    img_index = 0 
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    
    while idx < seq_len:
        if input_ids[idx] == processor.image_token_id and image_grid_thw is not None and img_index < len(image_grid_thw):
            grid = image_grid_thw[img_index]
            t_t = grid[0].item()
            h_t = grid[1].item() // merge_size
            w_t = grid[2].item() // merge_size
            
            img_len = t_t * h_t * w_t
            valid_len = min(img_len, seq_len - idx)
            
            t_grid = torch.arange(t_t).view(-1, 1, 1).expand(-1, h_t, w_t).flatten()
            h_grid = torch.arange(h_t).view(1, -1, 1).expand(t_t, -1, w_t).flatten()
            w_grid = torch.arange(w_t).view(1, 1, -1).expand(t_t, h_t, -1).flatten()
            
            positions_ids[0, idx : idx + valid_len] = step + t_grid[:valid_len]
            positions_ids[1, idx : idx + valid_len] = step + h_grid[:valid_len]
            positions_ids[2, idx : idx + valid_len] = step + w_grid[:valid_len]
            
            step += max(t_t, h_t, w_t)
            idx += valid_len
            img_index += 1
        else:
            positions_ids[0, idx] = step
            positions_ids[1, idx] = step
            positions_ids[2, idx] = step
            step += 1
            idx += 1
            
    return positions_ids

def main():
    parser = argparse.ArgumentParser(description="MiniMind-Qwen3VL 多模态推理测试")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm_stage2', type=str, help="权重名称前缀")
    
    # 路径参数
    parser.add_argument("--tokenizer_dir", type=str, default="./minimind_vl_tokenizer")
    parser.add_argument("--processor_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vision_config_path", type=str, default="./Qwen3_SigLIP2/vision_config.pt")
    parser.add_argument("--text_config_path", type=str, default="./minimind_vl_base_model/config.json")
    
    # 生成参数
    parser.add_argument('--max_new_tokens', default=1024, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数")
    parser.add_argument('--show_speed', default=1, type=int, choices=[0, 1])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float16', 'bfloat16'])
    args = parser.parse_args()
    
    model, processor = init_vlm_model(args)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    conversation = []
    
    print("\n" + "="*50)
    print("🚀 MiniMind-Qwen3VL 准备就绪！")
    print("="*50)

    while True:
        try:
            image_path = input("\n🖼️ 图片路径 (直接回车跳过即纯文本对话，输入 'quit' 退出): ").strip()
            if image_path.lower() == 'quit':
                break
                
            prompt = input("💬: ").strip()
            if not prompt:
                continue

            setup_seed(2026)
            
            # --- 历史管理 ---
            conversation = conversation[-args.historys:] if args.historys else []
            
            # --- 构建内容 ---
            images = []
            if image_path:
                try:
                    images.append(Image.open(image_path).convert("RGB"))
                    # 将图片占位符注入文本中
                    conversation.append({"role": "user", "content": f"<image>\n{prompt}"})
                except Exception as e:
                    print(f"⚠️ 图片加载失败 ({e})，将回退到纯文本模式。")
                    conversation.append({"role": "user", "content": prompt})
            else:
                conversation.append({"role": "user", "content": prompt})
            
            # --- 格式化 Prompt ---
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text = text.replace("<image>", processor.image_token)
            
            # --- 处理输入张量 ---
            process_kwargs = {
                "text": [text],
                "return_tensors": "pt",
            }
            if images:
                process_kwargs["images"] = images
                process_kwargs["max_pixels"] = 1024 * (28 ** 2) # 与 Dataset 保持一致

            inputs = processor(**process_kwargs)
            
            input_ids = inputs["input_ids"].squeeze(0)
            image_grid_thw = inputs.get("image_grid_thw")
            
            # --- 生成 3D 位置编码 ---
            positions_ids = generate_3d_positions_ids(input_ids, image_grid_thw, processor).unsqueeze(1).to(args.device)
            
            # --- 数据转移与类型转换 ---
            input_ids = inputs["input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)
            pixel_values = inputs.get("pixel_values")
            
            if pixel_values is not None:
                dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
                pixel_values = pixel_values.to(args.device, dtype=dtype)
                image_grid_thw = image_grid_thw.to(args.device)

            print('🤖: ', end='')
            st = time.time()
            
            # --- 执行生成 ---
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    positions_ids=positions_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    streamer=streamer,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=1.0
                )
            
            # --- 收尾与状态更新 ---
            response = processor.tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            conversation.append({"role": "assistant", "content": response})
            
            if args.show_speed:
                gen_tokens = len(generated_ids[0]) - len(input_ids[0])
                print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s')
            else:
                print()

        except KeyboardInterrupt:
            print("\n退出对话。")
            break
        except Exception as e:
            print(f"\n❌ 推理出错: {e}")

if __name__ == "__main__":
    main()