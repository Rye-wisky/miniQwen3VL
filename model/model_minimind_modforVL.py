# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig
from typing import List, Tuple, Optional, Union
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 768,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 12,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6403,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            mrope_section: List[int] = [12, 10, 10],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        self.mrope_section = mrope_section  # MRoPE分段配置，默认为[12, 10, 10]，代表T/H/W轴的维度分配


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

@dataclass
class MyCausalLMOutputWithPast(CausalLMOutputWithPast):
    # 显式添加 aux_loss 字段，这样 __init__ 就能识别它了
    aux_loss: Optional[torch.FloatTensor] = None

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

##修改
# --- 1. 核心 MRoPE 实现类 (集成 YaRN 外推) ---

class MiniMindMRoPE(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # 默认分段：例如 head_dim=64 (512//8), 则dim=head_dim//2=32, 分配为 T:12, H:10, W:10
        self.mrope_section = getattr(config, "mrope_section", [12, 10, 10])
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = getattr(config, "rope_theta", 1000000.0)
        
        #end
        self.end=config.max_position_embeddings
    
        # 基础频率计算
        dim = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 1).float() / dim))#[dim]
        self.attention_scaling = 1.0

        # --- YaRN 缩放逻辑 ---
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            # 获取 YaRN 参数
            original_max = rope_scaling.get("original_max_position_embeddings", 2048)
            factor = rope_scaling.get("factor", 16.0)
            beta_fast = rope_scaling.get("beta_fast", 32.0)
            beta_slow = rope_scaling.get("beta_slow", 1.0)
            self.attention_scaling = rope_scaling.get("attention_factor", 1.0)


            if self.end /original_max>1.0:#只有当外推长度超过原始最大位置编码长度时才应用 YaRN 频率修正
                # YaRN 频率修正公式
                # 计算受影响的维度范围
                def inv_dim_fn(b):
                    return (dim * math.log(original_max / (b * 2 * math.pi))) / (2 * math.log(self.rope_theta))
                
                low = max(math.floor(inv_dim_fn(beta_fast)), 0)
                high = min(math.ceil(inv_dim_fn(beta_slow)), dim - 1)
                
                # 生成插值系数 ramp
                if high != low:
                    ramp = torch.clamp((torch.arange(dim).float() - low) / (high - low), 0, 1)
                else:
                    ramp = torch.zeros(dim).float()
                    
                # 应用缩放：(1-ramp) 部分保持原样，ramp 部分乘以 1/factor
                # 实际上在 YaRN 中，频率越低对应的维度索引越大
                inv_freq = inv_freq * (1 - ramp + ramp / factor)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """将 [3, bs, sl, d] 的频率交错合并为 [bs, sl, d]"""
        freqs = freqs.contiguous()   # 【修改8】确保内存连续
        freqs_t = freqs[0].clone()   # 必须使用 clone 防止原地修改(inplace)导致异常
        for dim, offset in enumerate((1, 2), start=1):  # 依次处理 H 和 W 轴
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        position_ids: [3, batch, seq_len] 代表 T, H, W 坐标
        """
        # 1. 扩展维度以进行矩阵乘法
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # 2. 计算各轴旋转角度: [3, bs, seq_len, head_dim//2]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        
        # 3. 交错合并信息 (MRoPE 核心逻辑)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        
        # 4. 拼接并生成 cos/sin
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# --- 2. 辅助函数 ---

def rotate_half(x):
    """标准旋转操作"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """将旋转编码应用到 Q 和 K"""
    # 假设输入形状为 [bs, sl, heads, head_dim]
    cos = cos.unsqueeze(2) 
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """【修改1】适配标准KV shape: [bs, n_kv_heads, slen, head_dim]"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 【修改2】转置为标准 HF KV Cache 形状: [bsz, heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 【修改3】兼容 Transformers DynamicCache 对象和传统 Tuple 缓存
        # 【替换 Attention.forward 中处理 Cache 的逻辑】
        if past_key_value is not None:
            if hasattr(past_key_value, "update"): # 兼容新版 DynamicCache 对象
                xk, xv = past_key_value.update(xk, xv, self.layer_id)
                past_kv = past_key_value
            # 【修复点】不仅要是 tuple，且里面的元素不能是 None
            elif isinstance(past_key_value, tuple) and past_key_value[0] is not None: 
                xk = torch.cat([past_key_value[0], xk], dim=2)
                xv = torch.cat([past_key_value[1], xv], dim=2)
                past_kv = (xk, xv) if use_cache else None
            else:
                past_kv = (xk, xv) if use_cache else None
        else:
            past_kv = (xk, xv) if use_cache else None
        
        # repeat_kv 现已接收并返回 [bsz, heads, seq_len, head_dim]
        xk_rep = repeat_kv(xk, self.n_rep)
        xv_rep = repeat_kv(xv, self.n_rep)
        
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk_rep, xv_rep, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 注意这里: xk_rep.transpose(-2, -1) 变为 [bsz, n_heads, head_dim, kv_len]
            scores = (xq @ xk_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv_rep # [bsz, n_heads, seq_len, head_dim]

        # 恢复形状: [bsz, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        # 【修改4】注入 layer_id 给 attention，供 DynamicCache 使用
        self.self_attn.layer_id = layer_id
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None,
                visual_pos_masks=None, deepstack_visual_embed=None #修改for多模态
                ):  
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        
        #修改for多模态
        # --- DeepStack 视觉残差注入 ---
        if deepstack_visual_embed is not None and visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            deepstack_visual_embed = deepstack_visual_embed.to(hidden_states.dtype).to(hidden_states.device)
            local_hidden = hidden_states[visual_pos_masks, :].clone() + deepstack_visual_embed
            hidden_states[visual_pos_masks, :] = local_hidden
        # ------------------------------
        
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig,):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mrope_emb=MiniMindMRoPE(config)

    def forward(self, input_ids: Optional[torch.Tensor] = None, positions_ids: Optional[torch.Tensor] = None, 
                attention_mask=None, past_key_values=None, use_cache=False,
                inputs_embeds: Optional[torch.Tensor] = None, #修改for多模态：加入嵌入
                visual_pos_masks: Optional[torch.Tensor] = None,
                deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
                **kwargs):
    
    
        # 🚀 修复补丁：强制从 kwargs 中兜底获取，防止 Python 传参丢失
        if inputs_embeds is None and "inputs_embeds" in kwargs:
            inputs_embeds = kwargs["inputs_embeds"]
            
        # 修改for多模态支持从 inputs_embeds 直接输入
        if inputs_embeds is None:
            hidden_states = self.dropout(self.embed_tokens(input_ids))
        else:
            hidden_states = self.dropout(inputs_embeds)
        
        cos, sin = self.mrope_emb(hidden_states, positions_ids)
        position_embeddings = (cos, sin)

        # 【修复点】建立新的缓存收集器
        presents = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            # --- 智能分发 Cache ---
            if past_key_values is None:
                layer_past_key_value = None
            elif hasattr(past_key_values, "update"): 
                # 如果是 DynamicCache 对象，直接把原对象传给 Attention 让它自己调用 update
                layer_past_key_value = past_key_values
            else:
                # 如果是传统的 Tuple 列表，安全地取当前层
                layer_past_key_value = past_key_values[layer_idx] if layer_idx < len(past_key_values) else None

            #修改for多模态
            # 提取当前层的 DeepStack 特征
            layer_deepstack_embed = None
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                layer_deepstack_embed = deepstack_visual_embeds[layer_idx]
                
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=layer_past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            
            if presents is not None:
                presents.append(present)

        hidden_states = self.norm(hidden_states)

        # --- 智能返回 Cache ---
        if hasattr(past_key_values, "update"):
            next_cache = past_key_values # DynamicCache 自己维护了状态，直接返回它
        else:
            next_cache = tuple(presents) if presents is not None else None

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, next_cache, aux_loss # 注意这里返回 next_cache


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        
    # 【修改7】添加此方法，确保自回归时 positions_ids 长度与 input_ids 同步自增
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **kwargs
        )
        if "positions_ids" in model_kwargs and model_kwargs["positions_ids"] is not None:
            positions_ids = model_kwargs["positions_ids"]
            
            # 1. 拷贝最后一个 token 的 3D 坐标，形状: [3, batch, 1]
            new_pos = positions_ids[:, :, -1:].clone()
            
            #所有维度加1，因为训练的是文本decoder，保证T W H同步增长相当于一维RoPE
            new_pos += 1 
            
            # 3. 拼接到历史位置编码后面
            model_kwargs["positions_ids"] = torch.cat([positions_ids, new_pos], dim=-1)
            
        return model_kwargs
    
        # 👇 新增这个函数，用于适配 transformers 的 generate 方法
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        positions_ids = kwargs.get("positions_ids", None)
        
        # 【修复点】必须检查 cache 里面到底有没有存东西，不能只判断 is not None
        cache_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"): # DynamicCache API
                cache_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple) and len(past_key_values) > 0 and past_key_values[0] is not None and past_key_values[0][0] is not None:
                # 传统 Tuple 取张量长度 (dim=2 此时是 seq_len)
                cache_length = past_key_values[0][0].shape[2] 
                
        # 只有真正处于 Decode 阶段（拥有过去记忆），才把 input_ids 削减为 1
        if cache_length > 0:
            input_ids = input_ids[:, -1:]
            if positions_ids is not None:
                positions_ids = positions_ids[:, :, -1:]

        return {
            "input_ids": input_ids,
            "positions_ids": positions_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    # 同时也建议重写这个方法，以便 generate 正确处理 KV-Cache 的更新
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                positions_ids: Optional[torch.Tensor] = None,#[3, batch, seq_len]
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            positions_ids=positions_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MyCausalLMOutputWithPast(loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            aux_loss=aux_loss,
        )
        return output
