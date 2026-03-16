import torch
import os
import json
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

# ==========================================
# 第一步：安全扩充 Tokenizer
# ==========================================
tokenizer_path = "./" # 指向你现有的分词器目录
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 记录旧词表状态
old_vocab_size = len(tokenizer)
print(f"当前词表大小: {old_vocab_size}")

# 定义需要新增的特殊标记
new_special_tokens = [
    '<|im_start|>', 
    '<|im_end|>', 
    '<|vision_start|>', 
    '<|vision_end|>', 
    '<|image_pad|>'
]

# 检查是否已经存在，避免重复添加
existing_tokens = [t for t in new_special_tokens if t in tokenizer.get_vocab()]
tokens_to_add = [t for t in new_special_tokens if t not in tokenizer.get_vocab()]

if tokens_to_add:
    special_tokens_dict = {'additional_special_tokens': tokens_to_add}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"真正新增了 {num_added_toks} 个 token")
else:
    print("所有 Token 已存在，无需新增")

new_vocab_size = len(tokenizer)
print(f"扩充后词表大小: {new_vocab_size}")

# 【安全校验】确认 ID 确实是在末尾追加的
for token in new_special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id < old_vocab_size and token not in existing_tokens:
        raise ValueError(f"警告！Token {token} 的 ID ({token_id}) 侵入了原词表范围！")
    print(f"Token {token:15} ID: {token_id} (安全)")

# 保存新分词器
new_tokenizer_dir = "../minimind_vl_tokenizer"
tokenizer.save_pretrained(new_tokenizer_dir)

# ==========================================
# 第二步：调整模型权重并对齐
# ==========================================
config = MiniMindConfig()
config.tie_word_embeddings = True
model = MiniMindForCausalLM(config)

# 加载 .pth
ckpt_path = "../out/pretrain_768.pth"
state_dict = torch.load(ckpt_path, map_location='cpu')
if 'model' in state_dict:
    state_dict = state_dict['model']

# 允许非严格加载，此时 vocab_size 不匹配是正常的
model.load_state_dict(state_dict, strict=False)

# 调整 Embedding 矩阵大小
# 这会在权重矩阵的【末尾】增加新行，完全不会干扰原有的权重
model.resize_token_embeddings(new_vocab_size)

# 初始化新行
input_embeddings = model.get_input_embeddings().weight.data
output_embeddings = model.lm_head.weight.data

# 使用旧词表权重的均值初始化新行
# 注意：只在 old_vocab_size 之后的部分进行初始化
with torch.no_grad():
    mean_weight = input_embeddings[:old_vocab_size].mean(dim=0)
    for i in range(old_vocab_size, new_vocab_size):
        input_embeddings[i] = mean_weight
        output_embeddings[i] = mean_weight

# 绑定权重
model.tie_weights()

# ==========================================
# 第三步：保存
# ==========================================
save_dir = "../minimind_vl_base_model"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(save_dir, "minimind_vl_base.pth"))
config.vocab_size = new_vocab_size
config.save_pretrained(save_dir)

print(f"\n[成功] 基座已安全扩充。新 Token 的 ID 从 {old_vocab_size} 开始，未覆盖原有数据。")