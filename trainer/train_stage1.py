import os
import sys
import json
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset

# 导入你修改过的模型和数据集
from model.model_minimind_Qwen3VL import MiniQwen3VLConfig, MiniQwen3VLForCausalLM
from dataset.lm_dataset import miniQwen3VLDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, SkipBatchSampler

warnings.filterwarnings('ignore')
# 将项目根目录添加到 python 搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_model_loading(model, model_type="Model", check_trainable=None):
    """
    更加鲁棒的权重加载诊断脚本。
    model: 待检查的模型模块
    model_type: 描述文字
    check_trainable: 
        None -> 自动识别 (打印前3个含有 weight 的层)
        True -> 只检查可训练参数 (如 Projector)
        False -> 只检查已冻结参数 (如 Vision/LLM)
    """
    print(f"\n--- {model_type} 权重加载诊断 ---")
    
    # 处理 DDP 或 torch.compile 包装
    target_model = model
    if isinstance(target_model, DistributedDataParallel):
        target_model = target_model.module
    if hasattr(target_model, '_orig_mod'):
        target_model = target_model._orig_mod

    found_params = []
    all_params_count = 0
    
    # 遍历参数
    for name, param in target_model.named_parameters():
        all_params_count += 1
        # 基础筛选：必须是权重矩阵 (排除 bias 和 norm 的 offset 等)
        if "weight" not in name:
            continue
            
        # 根据需求筛选状态
        if check_trainable is True and not param.requires_grad:
            continue
        if check_trainable is False and param.requires_grad:
            continue
            
        found_params.append((name, param))
        if len(found_params) >= 3: 
            break

    # 如果按照指定状态 (冻结/可训练) 没找到，尝试放宽条件再找一次
    if not found_params:
        status_req = "可训练" if check_trainable else "冻结"
        print(f"提示: 在模块中未找到 [{status_req}] 状态的层 (总参数项: {all_params_count})。")
        print(f"正在尝试打印该模块前两个任意权重的状态以供排查...")
        
        fallback_count = 0
        for name, param in target_model.named_parameters():
            if "weight" in name:
                mu, std = param.data.mean().item(), param.data.std().item()
                st = "🔥可训练" if param.requires_grad else "❄️已冻结"
                print(f"  [Fallback] 层: {name:<40} | 状态: {st} | 均值: {mu:.6f} | Std: {std:.6f}")
                fallback_count += 1
                if fallback_count >= 2: break
        return

    # 正常打印
    for name, param in found_params:
        mu = param.data.mean().item()
        std = param.data.std().item()
        grad_status = "🔥可训练" if param.requires_grad else "❄️已冻结"
        print(f"层: {name:<50} | 状态: {grad_status} | 均值: {mu:.6f} | 标准差: {std:.6f}")

def custom_collate_fn(batch):
    """
    针对多模态数据定制的 collate_fn。
    因为 Qwen3-VL 的 pixel_values 针对不同尺寸的图片会展平为不同大小的张量，
    不能直接用 torch.stack，必须按第 0 维度拼接 (torch.cat)。
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    pixel_values_list = [item['pixel_values'] for item in batch if item['pixel_values'] is not None]
    image_grid_thw_list = [item['image_grid_thw'] for item in batch if item['image_grid_thw'] is not None]

    # 将 batch 内的图像特征拼接
    pixel_values = torch.cat(pixel_values_list, dim=0) if len(pixel_values_list) > 0 else None
    image_grid_thw = torch.cat(image_grid_thw_list, dim=0) if len(image_grid_thw_list) > 0 else None


    if 'positions_ids' in batch[0] and batch[0]['positions_ids'] is not None:
        positions_ids = torch.stack([item['positions_ids'] for item in batch]).transpose(0, 1).contiguous()
    else:
        positions_ids = None
        
    return {
        'input_ids': input_ids,
        'positions_ids': positions_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw
    }


def set_train_mode(model):
    """设置正确的训练/验证模式。冻结的模块应该处于 eval() 模式以免更新 BN/Dropout"""
    model.train()
    if hasattr(model, 'module'):
        model.module.visual.eval()
        model.module.language_model.eval()
    else:
        model.visual.eval()
        model.language_model.eval()


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, args=None, model=None, optimizer=None, scaler=None, autocast_ctx=None):
    start_time = time.time()
    set_train_mode(model) # 确保视觉和语言模型保持 eval，MLP保持 train

    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        positions_ids = batch.get('positions_ids').to(args.device)
        
        pixel_values = batch['pixel_values']
        image_grid_thw = batch['image_grid_thw']
        if pixel_values is not None:
            # 转换数据类型以匹配混合精度
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            pixel_values = pixel_values.to(args.device, dtype=dtype)
            image_grid_thw = image_grid_thw.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(
                input_ids=input_ids, 
                positions_ids=positions_ids,
                attention_mask=attention_mask,
                labels=labels, 
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
            loss = res.loss 
            if res.aux_loss is not None:
                loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 保存整个模型完整的 state_dict，方便后续阶段无缝加载
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model) # 兼容 torch.compile
            state_dict = raw_model.state_dict()
            
            ckp_path = f'{args.save_dir}/{args.save_weight}.pth'
            checkpoint = {
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'wandb_id': wandb.run.id if wandb else None
            }
            torch.save(checkpoint, ckp_path)
            Logger(f"已保存完整 Checkpoint 至 {ckp_path}")
            
            # 恢复训练模式
            set_train_mode(model)
            del state_dict, checkpoint

        del input_ids, labels, pixel_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-Qwen3VL Pretrain Phase 1 (Projector Alignment)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_vlm_stage1', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率 (对齐阶段LR通常较大)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=5000, help="模型保存间隔")
    
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练的最大截断长度")
    
    # 路径配置
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_i2t.parquet", help="训练数据路径")
    parser.add_argument("--tokenizer_dir", type=str, default="../minimind_vl_tokenizer", help="Tokenizer路径")
    parser.add_argument("--processor_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Processor名称/路径")
    
    parser.add_argument("--vision_config_path", type=str, default="../Qwen3_SigLIP2/vision_config.pt", help="视觉编码器Config加载路径")
    parser.add_argument("--vision_weight_path", type=str, default="../Qwen3_SigLIP2/vision_model_weights.bin", help="../Qwen3_SigLIP2/vision_model_weights.bin")
    parser.add_argument("--text_weight_path", type=str, default="../minimind_vl_base_model/minimind_vl_base.pth", help="../minimind_vl_base_model/minimind_vl_base.pth")
    parser.add_argument("--text_config_path", type=str, default="../minimind_vl_base_model/config.json", help="文本大模型(Decoder)的config.json加载路径")
    
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Pretrain-Stage1", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    # ========== 2. 检查 Checkpoint 状态 ==========
    # .pth 
    ckp_path = f'{args.save_dir}/{args.save_weight}.pth'
    ckp_data = None
    if args.from_resume == 1 and os.path.exists(ckp_path):
        if is_main_process(): Logger(f"找到 Checkpoint，准备从 {ckp_path} 续训...")
        ckp_data = torch.load(ckp_path, map_location='cpu')
    
    
    # ========== 2. 加载 Tokenizer 和 Processor (核心环节) ==========
    if is_main_process(): Logger(f"Loading tokenizer from {args.tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    
    if is_main_process(): Logger(f"Loading processor from {args.processor_name}")
    processor = AutoProcessor.from_pretrained(args.processor_name, trust_remote_code=True)
    
    # 【关键】注入自定义 ID 到 Processor
    processor.tokenizer = tokenizer
    processor.image_token_id = 6402  # 对应自定义词表中的 <|image_pad|>
    processor.vision_start_token_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    processor.vision_end_token_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    processor.video_token_id = None # 禁用视频

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置并初始化模型 ==========
    if is_main_process(): Logger(f"Loading vision config from {args.vision_config_path}")
    vision_config = torch.load(args.vision_config_path, weights_only=False)
    # 把 vision_config 直接塞进你的 MiniQwen3VLConfig，它内部会自动把它丢给 Qwen3VLVisionModel
     # 加载文本 Decoder 的 config
    text_config_dict = None
    if args.text_config_path and os.path.exists(args.text_config_path):
        if is_main_process(): Logger(f"Loading text config from {args.text_config_path}")
        with open(args.text_config_path, 'r', encoding='utf-8') as f:
            text_config_dict = json.load(f)
    else:
        if is_main_process(): Logger("未指定 text_config_path，将使用 MiniMind 默认参数构建 Decoder")
    config = MiniQwen3VLConfig(
        vision_config=vision_config, 
        text_config=text_config_dict, 
        image_token_id=processor.image_token_id
    )
    model = MiniQwen3VLForCausalLM(config).to(args.device)
    
    # 4.1 加载预训练权重
    if args.vision_weight_path:
        if is_main_process(): Logger(f"Loading Vision Encoder from {args.vision_weight_path}")
        model.visual.load_state_dict(torch.load(args.vision_weight_path, map_location='cpu', weights_only=False), strict=False)
        verify_model_loading(model.visual, "Vision Encoder", check_trainable=False)
        
    if args.text_weight_path:
        if is_main_process(): Logger(f"Loading Language Model from {args.text_weight_path}")
        model.language_model.load_state_dict(torch.load(args.text_weight_path, map_location='cpu'), strict=False)
        

    
    # 4.2 冻结核心参数，仅保留投影层训练（Stage 1 Alignment）
    for name, param in model.named_parameters():
        if "vision_proj" in name or "deepstack_proj" in name:
            param.requires_grad = True   # 解冻 MLP
        else:
            param.requires_grad = False  # 冻结 Vision & Text Encoder
    verify_model_loading(model.language_model, "Decoder LLM", check_trainable=False)
    if args.use_compile == 1:
        model.language_model = torch.compile(model.language_model)
        model.vision_proj = torch.compile(model.vision_proj)
        model.deepstack_proj = torch.compile(model.deepstack_proj)
        if is_main_process(): Logger('torch.compile enabled (LLM & Proj only)')
    
    verify_model_loading(model.vision_proj, "Vision Projector", check_trainable=True)    
    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        Logger(f"总参数量: {total_params / 1e6:.2f}M | 可训练参数量(MLP): {trainable_params / 1e6:.2f}M")
    
    # ========== 5. 数据集与 DataLoader ==========
    ds_raw = load_dataset('parquet', data_files=args.data_path, split='train')
    train_ds = miniQwen3VLDataset(ds_raw, processor, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 只将 require_grad=True 的参数（即我们刚刚解冻的 MLP）传入优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)
        
    
    
    # ========== 6. DDP包装 ==========
    if dist.is_initialized():
        # 如果有不可求导也不需同步的参数，可以加入到 ignore 集合
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # ========== 7. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume_mode = 'must' if wandb_id else None
        wandb_run_name = f"VLM-Stage1-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume_mode)

    # ========== 8. 开始训练 ==========
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        #跳过逻辑
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, 0)
        
        # 必须传入我们自定义的 collate_fn 才能正确处理 Qwen3 变长图像块
        loader = DataLoader(
            train_ds, 
            batch_sampler=batch_sampler, 
            num_workers=args.num_workers, 
            pin_memory=True, 
            collate_fn=custom_collate_fn
        )
        train_epoch(epoch, loader, len(loader), 0, wandb, args, model, optimizer, scaler, autocast_ctx)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()