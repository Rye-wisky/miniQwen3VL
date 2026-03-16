import os
import sys

# 设置包名并添加父目录到系统路径，以便正确导入同级目录下的自定义模块
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
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler

# 忽略警告信息
warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行一个训练周期的函数
    :param epoch: 当前轮数
    :param loader: 数据加载器
    :param iters: 总迭代次数
    :param start_step: 起始步数（用于断点续训）
    :param wandb: 实验日志记录工具
    """
    start_time = time.time()
    
    # enumerate(loader, start=...) 确保在断点续训时步数显示正确
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        # 数据移动到目标设备 (GPU)
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = pixel_values.to(args.device)

        # 1. 计算并更新学习率 (Cosine Annealing 策略)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 2. 前向传播（使用自动混合精度上下文）
        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            # 总损失 = 主损失 + MoE辅助损失 (如果开启了MoE)
            loss = res.loss + res.aux_loss
            # 梯度累积：将损失缩小，等效于平均梯度
            loss = loss / args.accumulation_steps

        # 3. 反向传播（使用Scaler处理float16梯度溢出问题）
        scaler.scale(loss).backward()

        # 4. 优化器更新（达到梯度累积步数时执行）
        if (step + 1) % args.accumulation_steps == 0:
            # 在更新前对梯度进行反缩放，用于梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器步进并更新缩放器
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度，set_to_none=True 能略微提高性能
            optimizer.zero_grad(set_to_none=True)

        # 5. 打印日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算剩余预计时间 (ETA)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, '
                   f'logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                   f'lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            
            if wandb: 
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, 
                          "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 6. 保存检查点 (仅在主进程执行)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            
            # 处理 DDP 包装的模型以提取原始状态字典
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            
            # 过滤掉视觉编码器的参数（通常是预训练好的，无需重复保存）以减小权重大小
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            # 转换为半精度并移动到CPU保存，节省显存和磁盘空间
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            
            torch.save(clean_state_dict, ckp)
            # 保存用于断点续训的完整检查点（包含优化器和缩放器状态）
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                           epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        # 显式手动释放大对象内存
        del input_ids, labels, pixel_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V 预训练脚本")
    # 路径与保存配置
    parser.add_argument("--save_dir", type=str, default="../out", help="模型导出目录")
    parser.add_argument('--save_weight', default='pretrain_vlm', type=str, help="保存权重的文件名前缀")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=4, help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="单卡 Batch Size")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="初始最大学习率")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪的最大范数")
    
    # 设备与精度配置
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="计算精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 线程数")
    
    # 间隔配置
    parser.add_argument("--log_interval", type=int, default=100, help="打印日志的间隔 Step")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存权重的间隔 Step")
    
    # 模型架构配置
    parser.add_argument('--hidden_size', default=512, type=int, help="LLM 隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="LLM 层数")
    parser.add_argument('--max_seq_len', default=640, type=int, help="最大文本/图文序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构")
    
    # 数据与加载配置
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_i2t.parquet", help="训练数据集路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="初始化权重来源 (llm/none/path)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测并尝试续训")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1], help="是否冻结语言模型部分（只训投影层）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否启用 torch.compile 加速")
    
    # 监控工具
    parser.add_argument("--use_wandb", action="store_true", help="是否启动训练监控 (SwanLab/Wandb)")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Pretrain", help="监控项目名称")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode() # DDP 多卡初始化
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 确保分布式训练下每张卡的初始种子不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 模型配置与检查点扫描 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, 
                           max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
    # 如果开启了 resume，尝试寻找最近的 checkpoint 记录
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度环境 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # autocast 会根据 dtype 自动选择部分算子运行在低精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验监控系统 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb # 这里使用了兼容 wandb 接口的 swanlab
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、分词器、处理器和数据集 ==========
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device, freeze_llm=bool(args.freeze_llm))
    
    # 使用 torch 2.x 的图编译功能，能提升 10%-20% 训练速度
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_ds = VLMDataset(args.data_path, tokenizer, preprocess=preprocess,
                          image_special_token=vlm_config.image_special_token,
                          max_length=vlm_config.max_seq_len)
    
    # 分布式采样器，确保每张卡分到的数据不重复
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 梯度缩放器，解决 float16 训练时的数值下溢问题
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 只对 requires_grad=True 的参数（即未冻结的参数）创建优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # ========== 6. 加载检查点状态 (Resume) ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 分布式数据并行 (DDP) 封装 ==========
    if dist.is_initialized():
        # 忽略某些不需要梯度同步的缓存参数（RoPE 的正余弦表通常是预计算的缓存）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 主训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 必须设置 sampler 的 epoch，否则每轮洗牌结果都一样
        if train_sampler: train_sampler.set_epoch(epoch)
        
        # 确定起始跳过的步数（用于准确续训）
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        # SkipBatchSampler 能够跳过已经训练过的 batch
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 训练结束，销毁进程组 ==========
    if dist.is_initialized(): dist.destroy_process_group()