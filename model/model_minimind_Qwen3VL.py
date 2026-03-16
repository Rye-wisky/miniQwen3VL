import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from torch.nn import functional as F
def square_root_reweighting_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    计算平方根重加权损失，用于平衡长序列多模态数据与短序列纯文本数据的权重。
    logits: [batch_size, seq_len, vocab_size]
    labels: [batch_size, seq_len]
    """
    batch_size = logits.size(0)
    
    # 1. 计算每个 Token 的独立 Loss (reduction='none')
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), 
        labels.reshape(-1), 
        ignore_index=ignore_index, 
        reduction='none'
    ).reshape(batch_size, -1)
    
    # 2. 生成有效 Token 的 Mask，并统计有效 Token 数量 (N)
    valid_mask = (labels != ignore_index).float()
    num_valid_tokens = valid_mask.sum(dim=-1) 
    
    # 防止 N 为 0 导致除零错误
    num_valid_tokens = torch.clamp(num_valid_tokens, min=1.0)
    
    # 3. 计算每个样本的平均 Loss
    loss_per_sample = (loss_per_token * valid_mask).sum(dim=-1) / num_valid_tokens
    
    # 4. 计算平方根权重 (\sqrt{N})
    weight_per_sample = torch.sqrt(num_valid_tokens)
    
    # 5. 加权平均得到最终 Loss
    total_weighted_loss = (loss_per_sample * weight_per_sample).sum()
    total_weight = weight_per_sample.sum()
    
    return total_weighted_loss / total_weight

class VisonProjMLP(nn.Module): 
    """对视觉编码器的输出进行下采样/投影以适配 Decoder 的 d_model"""
    def __init__(self, input_hidden_size, output_hidden_size):
        super().__init__()
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.intermediate_size = 3072 # 也可以设定为 output_hidden_size 的倍数
        
        self.gate_proj = nn.Linear(self.input_hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.input_hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# 1. 导入你修改过的 MiniMind 核心模型模块
from model.model_minimind_modforVL import MiniMindModel, MiniMindConfig, MyCausalLMOutputWithPast,MiniMindForCausalLM

# 2. 导入 Qwen3-VL 视觉模型 
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionConfig
from transformers import PretrainedConfig
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
class MiniQwen3VLConfig(MiniMindConfig):
    model_type = "mini_qwen3_vl"
    
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=6402, # 必须与你 custom_tokenizer 的 <|image_pad|> token ID 一致
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 实例化视觉配置
        if isinstance(vision_config, dict):
            self.vision_config = Qwen3VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen3VLVisionConfig()
        else:
            self.vision_config = vision_config
            
        # 实例化文本配置
        if isinstance(text_config, dict):
            self.text_config = MiniMindConfig(**text_config)
        elif text_config is None:
            self.text_config = MiniMindConfig()
        else:
            self.text_config = text_config
            
        self.image_token_id = image_token_id
        

class MiniQwen3VLForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniQwen3VLConfig
    
    def __init__(self, config: MiniQwen3VLConfig):
        super().__init__(config)
        self.config = config
        
        # 1. 初始化视觉编码器，并直接冻结参数 (Freeze)
        self.visual = Qwen3VLVisionModel(config.vision_config)
        for param in self.visual.parameters():
            param.requires_grad = False
            
        # 2. 挂载你的 MiniMind 文本大模型
        self.language_model = MiniMindModel(config.text_config)
        
        # =====================================================================
        # 3. 初始化视觉投影层 (Projectors) - 在此初始化！
        # =====================================================================
        # 获取视觉编码器的输出维度 (Qwen3-VL 通常在 out_hidden_size，也有直接在 hidden_size 的情况)
        vision_out_dim = getattr(config.vision_config, 'out_hidden_size', config.vision_config.hidden_size)
        text_dim = config.text_config.hidden_size # 你的大模型 d_model (如 768)
        
        # 实例化两个独立的投影器
        self.vision_proj = VisonProjMLP(input_hidden_size=vision_out_dim, output_hidden_size=text_dim)
        self.deepstack_proj = VisonProjMLP(input_hidden_size=vision_out_dim, output_hidden_size=text_dim)
        
        # 4. LM Head 词表输出头
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.language_model.embed_tokens.weight = self.lm_head.weight
        
    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value
        
    def get_placeholder_mask(self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.Tensor):
        """
        精确锁定文本序列中 <|image_pad|> 占位符的位置掩码。
        """
        if input_ids is not None:
            special_image_mask = (input_ids == self.config.image_token_id)
        else:
            image_token_embed = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token_embed).all(-1)

        n_image_tokens = special_image_mask.sum()
        if image_features is not None and n_image_tokens != image_features.shape[0]:
            raise ValueError(
                f"图像特征数量与占位符数量不匹配！文本占位符数量 {n_image_tokens}, 视觉特征数量 {image_features.shape[0]}"
            )
        return special_image_mask #shape: (batch_size, seq_len), dtype: torch.bool — 每个位置为 True 表示该 token 为 <|image_pad|>（用于在 inputs_embeds 中被视觉特征覆盖）

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,#batch 3 h w
        image_grid_thw: Optional[torch.Tensor] = None,#batch 3
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ):
        
        # 1. 将文本转化为基础词向量
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)# (batch_size, seq_len, embed_dim)
        else:
            inputs_embeds = kwargs.get("inputs_embeds", None)
            
        visual_pos_masks = None
        deepstack_visual_embeds = None
        
        # 2. 如果当前含有图像输入，进行特征融合
        if pixel_values is not None:
            # a) 视觉编码器前向传播，得到最后层输出与多层中间层 DeepStack 特征
            vision_outputs, deepstack_visual_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            
            if isinstance(vision_outputs, (list, tuple)):
                vision_outputs = torch.cat(vision_outputs, dim=0)
            vision_outputs = vision_outputs.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # =================================================================
            # 在此执行视觉特征投影 (Forward Pass)
            # =================================================================
            # 1. 主视觉特征投影 (从 2048 降维到 768)
            vision_outputs = self.vision_proj(vision_outputs)
            
            # 2. DeepStack 特征同样进行投影,但考虑到模型较小，只投影到第一层
            # if deepstack_visual_embeds is not None:
            #     deepstack_visual_embeds = [
            #         self.deepstack_proj(embed.to(inputs_embeds.device, inputs_embeds.dtype))
            #         for embed in deepstack_visual_embeds
            #     ]
            if deepstack_visual_embeds is not None:
                # 只取视觉编码器的第一层输出 (index 0)
                # 注意：这里包装成 List，因为你的 MiniMindModel 内部可能在循环里读取
                first_layer_feature = deepstack_visual_embeds[0].to(inputs_embeds.device, inputs_embeds.dtype)
                
                # 投影并只保留这一个特征
                deepstack_visual_embeds = [self.deepstack_proj(first_layer_feature)]
            
            # b) 定位占位符掩码
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, vision_outputs)
            visual_pos_masks = image_mask#batch seq
            
            #inputs_embeds batch seq d_model
            # c) 物理替换！把 inputs_embeds 里属于图片坑位的地方，用 vision_outputs 覆写

            #inputs_embeds batch seq d_model
            #image_mask batch seq
            #vision_outputs batch num_images d_model
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask] = vision_outputs.to(inputs_embeds.dtype)
        
        # 3. 将混合后的图文特征输入给 MiniMind 底层大语言模型
        # 注意：这里调用了修改后的 MiniMindModel，传入了 inputs_embeds 和 DeepStack 特征
        hidden_states, past_key_values, aux_loss = self.language_model(
            input_ids=None, # 置空，强制要求底层使用 inputs_embeds
            positions_ids=positions_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs
        )
        
        # 4. 计算 Logits 和交叉熵 Loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            #loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), 
            loss = square_root_reweighting_loss(shift_logits, shift_labels, ignore_index=-100)

        return MyCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            aux_loss=aux_loss,
        )

    # ---------------------------------------------------------
    # HF generate() 生成推理接口支持 (可选)
    # ---------------------------------------------------------
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **kwargs
        )
        if "positions_ids" in model_kwargs and model_kwargs["positions_ids"] is not None:
            positions_ids = model_kwargs["positions_ids"]
            new_pos = positions_ids[:, :, -1:].clone()
            # 1. 找到当前序列中所有维度里最大的那个索引值
            max_pos = torch.max(new_pos, dim=0, keepdim=True)[0] 
            # 2. 所有的维度都从这个最大值开始 +1
            new_pos = (max_pos + 1).expand_as(new_pos)
            model_kwargs["positions_ids"] = torch.cat([positions_ids, new_pos], dim=-1)
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        positions_ids = kwargs.get("positions_ids", None)
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        
        cache_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                cache_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple) and len(past_key_values) > 0 and past_key_values[0] is not None and past_key_values[0][0] is not None:
                cache_length = past_key_values[0][0].shape[2] 
                
        if cache_length > 0:
            input_ids = input_ids[:, -1:]
            if positions_ids is not None:
                positions_ids = positions_ids[:, :, -1:]
            # 阻断运算过的视觉特征重复向前传
            pixel_values = None
            image_grid_thw = None

        return {
            "input_ids": input_ids,
            "positions_ids": positions_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }

# =========================================================================
# 🚀 测试脚本：模拟前向传播
# =========================================================================
if __name__ == "__main__":
    import time
    print("="*60)
    print("🚀 开始测试 MiniMind-Qwen3-VL 架构")
    print("="*60)

    # --- 1. 初始化超小型配置用于快速测试 ---
    text_config = MiniMindConfig(
        hidden_size=768, 
        num_attention_heads=12, 
        num_hidden_layers=8, 
        vocab_size=6401 # 加了一个图片占位符
    )
    
    # 获取 Qwen3 默认的 vision_config (如果环境里没有，可以 mock 一下字典)
    try:
        vision_config = Qwen3VLVisionConfig(hidden_size=1152, depth=2)
    except:
        vision_config = {"hidden_size": 1152, "depth": 2}
        
    config = MiniQwen3VLConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=6403
    )
    
    # 实例化大模型
    model = MiniQwen3VLForCausalLM(config)
    model.eval() # 切换到评估模式

    # ---------------------------------------------------------------------
    # 💡 核心 Mock 操作 (仅为测试，避免复杂的 Image Processor 报错)
    # 真实的 Qwen3VisionModel 需要极其特殊的 grid_thw 和 pixel_values 计算逻辑。
    # 为了纯粹测试你的图文对齐/替换、Loss计算和推演逻辑，我们暂时接管它的输出。
    # 我们假设当前图片编码后会输出 4 个 Token。
    # ---------------------------------------------------------------------
    NUM_IMG_TOKENS = 4
    VISION_DIM = getattr(config.vision_config, 'out_hidden_size', config.vision_config.hidden_size)
    
    def dummy_vision_forward(pixel_values, grid_thw=None):
        # 模拟主输出 [1, num_img_tokens, vision_dim]
        out = [torch.randn(NUM_IMG_TOKENS, VISION_DIM)]
        # 模拟 DeepStack 跨层输出
        deepstack = [torch.randn(NUM_IMG_TOKENS, VISION_DIM) for _ in range(config.text_config.num_hidden_layers)]
        return out, deepstack
    
    # 临时替换
    model.visual.forward = dummy_vision_forward

    # --- 2. 准备张量数据 ---
    # 批次 1，总长度 10
    # 模拟文本：[BOS, "图", "片", <|image_pad|>*4, "很", "美", EOS]
    # 注意：<|image_pad|> (6400) 的数量必须等于 NUM_IMG_TOKENS (4)
    input_ids = torch.tensor([[1, 233, 344, 6400, 6400, 6400, 6400, 455, 566, 2]], dtype=torch.long)
    batch_size, seq_len = input_ids.shape

    # 模拟图片张量 (随便给个形状，因为上面 mock 掉了，但占位需要)
    pixel_values = torch.randn(1, 3, 224, 224) 
    image_grid_thw = torch.tensor([[1, 2, 2]]) # T=1, H=2, W=2 -> 4 patches

    # 构建 3D MRoPE positions_ids [3, batch, seq_len]
    # 我们为 <|image_pad|> 赋予特殊的 H, W 坐标，其余文本为 1D 线性自增
    positions_t = torch.tensor([[0, 1, 2, 3, 3, 3, 3, 4, 5, 6]]) # 时间轴(文本)
    positions_h = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]) # 高度轴(图片)
    positions_w = torch.tensor([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0]]) # 宽度轴(图片)
    positions_ids = torch.stack([positions_t, positions_h, positions_w], dim=0) # Shape: [3, 1, 10]

    # 训练用的 Labels (模拟 Next Token Prediction)
    labels = input_ids.clone()

    print("\n" + "-"*40)
    print("🔥 场景 1: 模拟训练 (Training Forward Pass)")
    print("-"*40)
    
    start_t = time.time()
    outputs = model(
        input_ids=input_ids,
        positions_ids=positions_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=labels
    )
    print(f"✅ 前向传播成功！耗时: {time.time() - start_t:.4f}s")
    print(f"📊 Logits 形状: {outputs.logits.shape} (预期: [1, 10, 6400])")
    print(f"📉 Loss: {outputs.loss.item():.4f}")
    if outputs.aux_loss is not None and isinstance(outputs.aux_loss, torch.Tensor):
        print(f"🔧 Aux Loss (MoE): {outputs.aux_loss.item():.4f}")


    print("\n" + "-"*40)
    print("⚡ 场景 2: 模拟推理 (Inference / Generation) 与 positions_ids 追踪")
    print("-"*40)
    
    with torch.no_grad():
        # Step A: Prefill (预填充阶段) - 传入完整文本和图片，生成 KV Cache
        print("[Step A] Prefill 阶段: 消化 prompt 和图片...")
        prefill_outputs = model(
            input_ids=input_ids,
            positions_ids=positions_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True
        )
        
        # 提取 KV Cache
        past_key_values = prefill_outputs.past_key_values
        print(f"✅ Prefill 完成，获取 KV Cache 长度: {past_key_values[0][0].shape[2]}")
        
        # 模拟 transformers generate 循环内部的变量状态
        model_kwargs = {
            "positions_ids": positions_ids,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "use_cache": True,
            "cache_position": torch.arange(input_ids.shape[1], device=input_ids.device)
        }
        
        # 假设通过 argmax 预测出了下一个 Token ID 为 888
        next_token_id = torch.tensor([[888]], dtype=torch.long)
        
        # Step B: Decode (解码阶段) - 逐字生成
        print("\n[Step B] Decode 阶段: 自回归生成")
        for step in range(1, 4): # 模拟生成 3 个 Token
            print(f"\n--- 生成步数: {step} ---")
            
            # 1. 更新 kwargs (核心观察点：positions_ids 的递增)
            model_kwargs = model._update_model_kwargs_for_generation(prefill_outputs, model_kwargs)
            current_pos_ids = model_kwargs["positions_ids"]
            
            print(f"当前整体 positions_ids 形状: {current_pos_ids.shape}")
            print("🚀 新增的 3D Position ID (T, H, W):")
            print(f"T: {current_pos_ids[0, 0, -1].item()}")
            print(f"H: {current_pos_ids[1, 0, -1].item()}")
            print(f"W: {current_pos_ids[2, 0, -1].item()}")
            
            # 2. 准备 inputs (将只切片保留最后一个 token 和 pos_id，切断图片重复计算)
            model_inputs = model.prepare_inputs_for_generation(next_token_id, **model_kwargs)
            
            print(f"Prepare后输入 Input IDs 形状: {model_inputs['input_ids'].shape}")
            print(f"Prepare后输入 Pos IDs 形状:   {model_inputs['positions_ids'].shape}")
            print(f"图片特征是否已被清空(避免重算): {'pixel_values' not in model_inputs or model_inputs['pixel_values'] is None}")
            
            # 3. 前向传播
            prefill_outputs = model(**model_inputs)
            
            # 更新历史变量以便下一步循环
            model_kwargs["past_key_values"] = prefill_outputs.past_key_values
            next_token_id += 1 # 随便改个数字模拟新 token
            
    print("\n🎉 测试圆满结束！多模态逻辑完全闭环。")