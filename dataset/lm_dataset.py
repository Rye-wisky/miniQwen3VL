from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # --- 新增：生成 MRoPE 所需的 3D position_ids [3, max_length] ---
        # T 轴为正常的 0, 1, 2... 序列
        t_ids = torch.arange(self.max_length, dtype=torch.long)
        h_ids = torch.arange(self.max_length, dtype=torch.long)
        w_ids = torch.arange(self.max_length, dtype=torch.long)
        # 堆叠成 [3, seq_len]
        pos_ids = torch.stack([t_ids, h_ids, w_ids])
        
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels, pos_ids


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }

import io
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
import pandas as pd
# ==========================================
# 1. 环境配置
# ==========================================
TOKENIZER_DIR = "../minimind_vl_tokenizer"
PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct" 
DATASET_PATH = "./pretrain_i2t.parquet" 

MAX_TOKENS = 1024
MAX_PIXELS = MAX_TOKENS * (28 ** 2)



# ==========================================
# 2. 多模态数据集类 
# ==========================================
class miniQwen3VLDataset(Dataset):
    def __init__(self, dataset, processor, max_length=1024, debug=False):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.debug = debug
        
        # 预先获取核心 ID，避免在 __getitem__ 中重复计算
        self.image_token_id = processor.image_token_id
        self.im_start_id = 1 # 你的 im_start / bos
        self.im_end_id = 2   # 你的 im_end / eos
        
        # 预先编码前缀，用于匹配
        # 注意：这里需要根据你的分词器实际输出调整
        self.assistant_prefix = self.processor.tokenizer.encode("assistant\n", add_special_tokens=False)
        self.target_prefix = [self.im_start_id] + self.assistant_prefix
    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def debug_labels(input_ids, labels, tokenizer):
        """
        绿色显示计算 Loss 的 Token，红色显示被屏蔽的 Token
        """
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print("\n--- 标签屏蔽可视化 (绿色为计算 Loss) ---")
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # 替换特殊空格字符以便阅读
            token = token.replace('Ġ', ' ')
            
            if label == -100:
                # 屏蔽部分：用红色打印
                print(f"\033[91m{token}\033[0m", end=" ")
            else:
                # Loss 部分：用绿色加粗打印
                print(f"\033[1;92m{token}\033[0m", end=" ")
            
            # 每 10 个 token 换行，方便观察
            if (i + 1) % 10 == 0:
                print()
        print("\n" + "-"*40)
        
    def __getitem__(self, index):
        row = self.dataset[index]
        conversations = row['conversations']
        if isinstance(conversations, str):
            conversations = json.loads(conversations)

        # 处理图像
        image_bytes_list = row.get('image_bytes', [])
        images = []
        if image_bytes_list:
            images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]

        # 格式化文本
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        text = text.replace("<image>", self.processor.image_token)

        # 处理输入
        process_kwargs = {
            "text": [text],
            "padding": "max_length",
            "max_length": self.max_length,
            "truncation": True, # 建议开启，防止单条超长数据拖垮显存
            "return_tensors": "pt"
        }
        if images:
            process_kwargs["images"] = images
            process_kwargs["max_pixels"] = MAX_PIXELS

        inputs = self.processor(**process_kwargs)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # 构造 Labels
        labels = torch.full_like(input_ids, -100)
        
        # 寻找 Assistant 区间
        i = 0
        while i < len(input_ids):
            # 匹配前缀 [1, assistant_id, newline_id]
            if input_ids[i : i + len(self.target_prefix)].tolist() == self.target_prefix:
                start_res = i + len(self.target_prefix)
                # 寻找对应的结束符 im_end (2)
                end_res = start_res
                while end_res < len(input_ids) and input_ids[end_res] != self.im_end_id:
                    end_res += 1
                
                # 填充真正的标签（包含结尾的 im_end）
                labels[start_res : end_res + 1] = input_ids[start_res : end_res + 1]
                i = end_res + 1
            else:
                i += 1

        # 视觉 Token 强制屏蔽 (双重保险)
        vision_ids = {self.image_token_id, 
                      getattr(self.processor, "vision_start_token_id", None),
                      getattr(self.processor, "vision_end_token_id", None)}
        
        for v_id in vision_ids:
            if v_id is not None:
                labels[input_ids == v_id] = -100

        # 屏蔽 Padding 部分
        labels[attention_mask == 0] = -100

        # -------------------------------------------------------------
        # 新增：手动计算 3D Position IDs (MRoPE 核心逻辑)
        # -------------------------------------------------------------
        positions_ids = torch.zeros(3, self.max_length, dtype=torch.long)
        step = 0 
        idx = 0  
        img_index = 0 
        
        # 获取图像的 merge_size（Qwen3 默认是 2）
        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        image_grid_thw = inputs.get("image_grid_thw")

        while idx < self.max_length:
            if input_ids[idx] == self.image_token_id and image_grid_thw is not None and img_index < len(image_grid_thw):
                # 获取当前图片的 T, H, W 原始 patch 数量
                grid = image_grid_thw[img_index]
                t_t = grid[0].item()
                h_t = grid[1].item() // merge_size
                w_t = grid[2].item() // merge_size
                
                img_len = t_t * h_t * w_t
                valid_len = min(img_len, self.max_length - idx) # 防止溢出 max_length
                
                # 生成当前图片的局部 3D 坐标网格
                t_grid = torch.arange(t_t).view(-1, 1, 1).expand(-1, h_t, w_t).flatten()
                h_grid = torch.arange(h_t).view(1, -1, 1).expand(t_t, -1, w_t).flatten()
                w_grid = torch.arange(w_t).view(1, 1, -1).expand(t_t, h_t, -1).flatten()
                
                # 将局部坐标加上基础 step，赋值给全局 positions_ids
                positions_ids[0, idx : idx + valid_len] = step + t_grid[:valid_len]
                positions_ids[1, idx : idx + valid_len] = step + h_grid[:valid_len]
                positions_ids[2, idx : idx + valid_len] = step + w_grid[:valid_len]
                
                # Qwen-VL 官方逻辑：图片块处理完后，基础步长前进 3D 网格中的最大维度
                step += max(t_t, h_t, w_t)
                idx += valid_len
                img_index += 1
            else:
                # 处理普通文本 Token 或者 Padding
                positions_ids[0, idx] = step
                positions_ids[1, idx] = step
                positions_ids[2, idx] = step
                
                if attention_mask[idx] == 1:
                    step += 1 # 只有非 padding 文本步长才自增
                idx += 1
        
        if self.debug:
            self.debug_labels(input_ids, labels, self.processor.tokenizer)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels,
            "positions_ids": positions_ids 
        }


# 在 Dataset 的 __getitem__ 结尾或 main 函数中调用：
# 

# ==========================================
# 3. 验证逻辑
# ==========================================
def main():
    # 初始化组件
    print(f"正在加载处理器: {PROCESSOR_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME, trust_remote_code=True)
    processor.tokenizer = tokenizer 
    processor.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    # Qwen3-VL 还有两个标志视觉区间的 token，如果你的词表里有，也得对上
    # 如果没有，processor 默认会去找 <|vision_start|>
    # 强制同步自定义 ID
    processor.image_token_id = 6402
    processor.vision_start_token_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    processor.vision_end_token_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    # 禁用视频逻辑（防止它去寻找不存在的 <|video_pad|>）
    processor.video_token_id = None

    print(f"已同步 ID - Image Token ID: {processor.image_token_id}")
    print(f"已更换自定义分词器")
    
    # 加载 Parquet 数据集
    # 运行这个测试
    print(f"Image Token: {processor.image_token}")
    print(f"Token ID: {processor.tokenizer.convert_tokens_to_ids(processor.image_token)}")
    print(f"正在加载数据集: {DATASET_PATH}")
    ds = load_dataset("parquet", data_files=DATASET_PATH, split="train")
    # 3. 实例化自定义 Dataset
    my_dataset = miniQwen3VLDataset(ds, processor, max_length=MAX_TOKENS)
    
    # 4. 取出一个含图片的样本进行验证
    # 找到第一个包含图片的索引
    sample_idx = 0
    for i in range(len(ds)):
        if ds[i].get('image_bytes'):
            sample_idx = i
            break
            
    sample = my_dataset[sample_idx]
    
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    pixel_values = sample["pixel_values"]
    grid_thw = sample["image_grid_thw"]

    print("\n" + "="*50)
    print(" 验证结果分析 ")
    print("="*50)

    # --- 验证 A: 图像占位符展开 ---
    img_token_count = (input_ids == processor.image_token_id).sum().item()
    print(f"1. 图像 Token 展开验证:")
    print(f"   - 预设 Image Token ID: {processor.image_token_id}")
    print(f"   - Input IDs 中包含的 Image Token 数量: {img_token_count}")
    
    if grid_thw is not None:
        # Qwen3-VL 逻辑: tokens = (H * W) // (merge_size^2)
        # 这里 grid_thw[0] 是 [T, H, W]
        expected_tokens = (grid_thw[0][1] * grid_thw[0][2]) // (processor.image_processor.merge_size**2)
        print(f"   - 根据 grid_thw 计算出的预期 Token 数: {expected_tokens}")
        if img_token_count == expected_tokens:
            print("   ✅ [PASS] 图像占位符已成功根据图片尺寸动态展开。")
        else:
            print("   ❌ [FAIL] 占位符数量与图片特征尺寸不符。")

    # --- 验证 B: Loss 设计 (Labels 屏蔽) ---
    print(f"\n2. Loss Mask (Labels) 验证:")
    
    # 统计信息
    total_len = len(input_ids)
    active_loss_tokens = (labels != -100).sum().item()
    print(f"   - 总 Token 长度: {total_len}")
    print(f"   - 参与 Loss 计算的 Token 数: {active_loss_tokens}")

    # 可视化前 50 个参与计算的 Token
    print(f"\n3. 抽样查看 Loss 计算区间 (Token -> Label):")
    decoded_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    
    found_active = 0
    for i in range(total_len):
        token_name = decoded_tokens[i]
        label_val = labels[i].item()
        
        # 重点检查图像 token 处是否为 -100
        if input_ids[i] == processor.image_token_id:
            if label_val != -100:
                print(f"   ❌ [ERR] 图像 Token '{token_name}' (idx:{i}) 未被屏蔽！")
        
        # 打印一部分 active 的 label 看看是不是真正的回复内容
        if label_val != -100 and found_active < 15:
            print(f"   [Loss Active] Pos {i}: '{token_name}' -> Label ID: {label_val}")
            found_active += 1
            
    # --- 验证 C: 边界检查 ---
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id in input_ids:
        indices = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
        all_passed = True
        for idx in indices:
            if labels[idx] == -100:
                # 注意：根据你的逻辑，im_end 应该保留以便模型学习停止
                print(f"   ⚠️ [INFO] <|im_end|> 在位置 {idx} 被屏蔽了（取决于你的训练策略）")
                all_passed = False
        if all_passed:
            print("   ✅ [PASS] <|im_end|> 已保留，模型将学习停止符。")

    print("="*50 + "\n")
if __name__ == "__main__":
    main()