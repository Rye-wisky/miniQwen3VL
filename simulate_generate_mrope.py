import torch

def simulate_mrope_generation():
    # 模拟参数
    batch_size = 1
    prompt_len = 3
    max_new_tokens = 2
    device = "cpu"

    # 1. 初始输入 (Prompt)
    # 假设输入是: [BOS, "你好", "啊"] -> token_id: [1, 10, 20]
    input_ids = torch.tensor([[1, 10, 20]], device=device)
    
    print("--- 阶段 1: Prefill (预填充阶段) ---")
    # 构造初始 3D 位置: T 轴是 0, 1, 2; H 和 W 轴是 0
    seq_len = input_ids.size(1)
    t_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0) # [1, 1, 3]
    hw_ids = torch.zeros_like(t_ids) # [1, 1, 3]
    positions_ids = torch.cat([t_ids, hw_ids, hw_ids], dim=0) # [3, 1, 3]
    
    print(f"输入 Input IDs 形状: {input_ids.shape} -> 内容: {input_ids.tolist()}")
    print(f"输入 Positions IDs (T轴): {positions_ids[0].tolist()}")
    print(f"输入 Positions IDs (H轴): {positions_ids[1].tolist()}")
    
    # 模拟进入 generate 循环
    past_key_values = "dummy_cache_after_prefill" # 模拟缓存已建立
    
    print("\n--- 阶段 2: Decoding (增量生成阶段 - 第1个新Token) ---")
    # 假设预测出了下一个词 "呢" -> token_id: 30
    # 此时 input_ids 会变成 [1, 10, 20, 30]
    input_ids = torch.cat([input_ids, torch.tensor([[30]])], dim=-1)
    
    # 【核心逻辑】动态计算递增的 T 轴
    # 在 KV-Cache 模式下，我们只需要当前最后一个位置的 ID
    # 当前总长度是 4，所以最后一个 T 坐标应该是 3 (0-indexed)
    current_total_len = input_ids.shape[1] 
    last_t_index = current_total_len - 1
    
    # 构造当前步的 3D ID: [3, 1, 1]
    curr_input_ids = input_ids[:, -1:] # 只取最后一个词 [1, 1]
    curr_pos_ids = torch.tensor([
        [[last_t_index]], # T 轴: 3
        [[0]],            # H 轴: 0
        [[0]]             # W 轴: 0
    ])
    
    print(f"输入 Input ID (仅当前): {curr_input_ids.tolist()}")
    print(f"输入 Positions IDs (T轴): {curr_pos_ids[0].item()}")
    print("注意：模型会拿着这个 T=3 的位置去缓存里找前 0,1,2 的 K/V 做计算")

    print("\n--- 阶段 3: Decoding (增量生成阶段 - 第2个新Token) ---")
    # 假设又预测出了下一个词 "。" -> token_id: 2
    input_ids = torch.cat([input_ids, torch.tensor([[2]])], dim=-1)
    
    # 再次计算
    current_total_len = input_ids.shape[1] 
    last_t_index = current_total_len - 1
    curr_input_ids = input_ids[:, -1:] # [1, 1]
    curr_pos_ids = torch.tensor([[[last_t_index]], [[0]], [[0]]]) # T 轴变为 4
    
    print(f"输入 Input ID (仅当前): {curr_input_ids.tolist()}")
    print(f"输入 Positions IDs (T轴): {curr_pos_ids[0].item()}")

if __name__ == "__main__":
    simulate_mrope_generation()