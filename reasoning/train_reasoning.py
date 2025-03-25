import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import time
import logging
import argparse
from functools import partial
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reasoning_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("reasoning_transformer_training")

# Import local modules
from transformer.params import ModelConfig
from transformer.transformer import ReasoningTransformer
from data.reasoning_data import get_reasoning_data
from tokenization.custom_tokenizer import get_custom_tokenizer
from tokenization.dataset_tokenize import tokenize_reasoning_dataset


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, warmup_iters, max_iters):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.current_iter = 0
        
    def step(self):
        # Linear warmup
        if self.current_iter < self.warmup_iters:
            lr_scale = min(1.0, float(self.current_iter + 1) / self.warmup_iters)
        # Cosine decay phase
        else:
            progress = float(self.current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
        
        self.current_iter += 1
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class ReasoningDataset(Dataset):
    """Dataset for reasoning tasks with prompt and response pairs"""
    
    def __init__(self, tokenized_data, block_size):
        self.prompt_tokens = tokenized_data['prompt_tokens']
        self.response_tokens = tokenized_data['response_tokens']
        self.block_size = block_size
    
    def __len__(self):
        return len(self.prompt_tokens)
    
    def __getitem__(self, idx):
        prompt_tokens = self.prompt_tokens[idx]
        response_tokens = self.response_tokens[idx]
        
        # Combine prompt and response tokens with special tokens
        # BOS is assumed to be added during tokenization if needed
        combined_tokens = prompt_tokens + response_tokens
        
        # Truncate or pad to block_size
        if len(combined_tokens) > self.block_size:
            combined_tokens = combined_tokens[:self.block_size]
        else:
            combined_tokens = combined_tokens + [-1] * (self.block_size - len(combined_tokens))
        
        # Create input and target tensors
        x = torch.tensor(combined_tokens, dtype=torch.long)
        
        # Create targets, shifting by 1
        y = torch.full_like(x, -1)  # Fill with ignore index
        
        # Set target for prompt tokens to -1 (we don't want to predict prompt)
        prompt_len = min(len(prompt_tokens), self.block_size)
        response_len = min(len(response_tokens), self.block_size - prompt_len)
        
        # For response tokens, target is the next token
        if response_len > 0:
            response_range = slice(prompt_len, prompt_len + response_len - 1)
            y[response_range] = x[prompt_len + 1:prompt_len + response_len]
        
        # Create attention mask (attending only to tokens up to current position)
        attention_mask = torch.tril(torch.ones(self.block_size, self.block_size))
        
        return x, y, attention_mask


@torch.no_grad()
def estimate_loss(model, dataloaders, eval_iters):
    """Estimate loss on datasets"""
    model.eval()
    losses = {}
    
    for split, dataloader in dataloaders.items():
        losses[split] = []
        
        # Use a limited number of batches for evaluation
        for i, (x, y, mask) in enumerate(dataloader):
            if i >= eval_iters:
                break
                
            x, y, mask = x.to(model.device), y.to(model.device), mask.to(model.device)
            
            # Compute loss
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(x, targets=y, attention_mask=mask)
            
            losses[split].append(loss.item())
    
    model.train()
    
    # Average losses
    avg_losses = {split: np.mean(split_losses) if split_losses else 0.0 
                for split, split_losses in losses.items()}
    
    return avg_losses


def train(rank, world_size, config):
    """Main training function for distributed training"""
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directories
    if rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.log_dir)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = get_custom_tokenizer(
        tokenizer_type=config.tokenizer_type,
        vocab_size=config.vocab_size,
        model_path=config.tokenizer_path
    )
    
    # Load and tokenize dataset
    logger.info("Loading reasoning dataset...")
    raw_dataset = get_reasoning_data()
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = tokenize_reasoning_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        num_cores=torch.get_num_threads() // world_size
    )
    
    # Split dataset into train/val
    train_size = int(0.95 * len(tokenized_dataset))
    val_size = len(tokenized_dataset) - train_size
    
    train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(
        tokenized_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Create datasets
    train_dataset = ReasoningDataset(train_dataset_raw, config.block_size)
    val_dataset = ReasoningDataset(val_dataset_raw, config.block_size)
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Create model
    logger.info("Creating model...")
    model = ReasoningTransformer(
        vocab_size=tokenizer.n_vocab,
        embed_dim=config.n_embd,
        num_heads=config.n_head,
        num_layers=config.n_layer,
        max_seq_len=config.block_size,
        dropout_prob=config.dropout,
        use_gradient_checkpoint=config.gradient_checkpointing,
        use_flash_attn=config.use_flash_attn,
        use_rope=config.use_rope,
        use_geglu=True
    )
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(config.checkpoint_dir, 'latest_checkpoint.pt')
    start_iter = 0
    best_val_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_iter = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resuming from iteration {start_iter} with validation loss {best_val_loss}")
    
    # Move model to device
    model = model.to(device)
    model.device = device
    
    # Enable mixed precision if requested
    dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    
    # Wrap model with DDP
    model = DDP(
        model, 
        device_ids=[rank], 
        output_device=rank, 
        find_unused_parameters=False,
        broadcast_buffers=False
    )
    
    # Optimizer with weight decay fix
    # Split parameters into weight decay and no weight decay groups
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in ['bias', 'layer_norm', 'embedding']) and p.requires_grad],
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in ['bias', 'layer_norm', 'embedding']) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    
    # Set initial learning rate for scheduler
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = config.learning_rate
    
    # Create learning rate scheduler
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_iters, config.max_iters)
    
    # Load optimizer state if resuming
    if os.path.exists(checkpoint_path) and start_iter > 0:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Update scheduler's current iteration
        scheduler.current_iter = start_iter
    
    # Create gradient scaler for mixed precision
    scaler = torch.amp.GradScaler()
    
    # Load scaler state if resuming
    if os.path.exists(checkpoint_path) and start_iter > 0 and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    logger.info(f"Starting training from iteration {start_iter}")
    iter_num = start_iter
    
    # Get total iterations and batches per epoch
    if rank == 0:
        print(f"Total iterations: {config.max_iters}")
        print(f"Batches per epoch: {len(train_loader)}")
    
    tokens_processed = 0
    train_iter = iter(train_loader)
    
    model.train()
    for iter_num in range(start_iter, config.max_iters):
        # Update sampler for each epoch
        if iter_num % len(train_loader) == 0:
            train_sampler.set_epoch(iter_num // len(train_loader))
        
        # Get batch
        try:
            x, y, attention_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y, attention_mask = next(train_iter)
        
        # Move to device
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            logits, loss = model(x, targets=y, attention_mask=attention_mask)
        
        # normalize loss by accumulation steps
        loss_value = loss.item()
        loss = loss / config.accumulation_steps
        
        # backward pass with scaled loss
        scaler.scale(loss).backward()
        
        if (iter_num + 1) % config.accumulation_steps == 0:
            # clip gradients for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # step optimizer with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            
            # step scheduler
            scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        tokens_processed += config.batch_size * config.block_size * world_size
        
        if rank == 0 and iter_num % 10 == 0:
            lr = scheduler.get_lr()
            iter_time = (time.time() - start_time) / (iter_num - start_iter + 1)
            tokens_per_sec = config.batch_size * config.block_size * world_size / iter_time
            
            print(f"Iter {iter_num}: loss {loss_value:.4f}, lr {lr:.6f}, {tokens_per_sec:.2f} tokens/sec")
            
            writer.add_scalar('training/loss', loss_value, iter_num)
            writer.add_scalar('training/learning_rate', lr, iter_num)
            writer.add_scalar('training/tokens_per_sec', tokens_per_sec, iter_num)
            writer.add_scalar('training/tokens_processed', tokens_processed, iter_num)
        
        if (rank == 0 and 
            (iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1)):
            loss_dict = estimate_loss(model, dataloaders, config.eval_iters)
            
            print(f"Iter {iter_num}: train loss {loss_dict['train']:.4f}, val loss {loss_dict['val']:.4f}")
            
            for split, loss_val in loss_dict.items():
                writer.add_scalar(f'evaluation/{split}_loss', loss_val, iter_num)
            
            if loss_dict['val'] < best_val_loss:
                best_val_loss = loss_dict['val']
                
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': vars(config)
                }
                
                torch.save(checkpoint, os.path.join(config.checkpoint_dir, 'best_model.pt'))
                print(f"New best model saved with val loss: {best_val_loss:.4f}")
            
            if iter_num % (config.eval_interval * 5) == 0:
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'iter_num': iter_num,
                    'val_loss': loss_dict['val'],
                    'config': vars(config)
                }
                
                torch.save(checkpoint, os.path.join(config.checkpoint_dir, f'checkpoint_{iter_num}.pt'))
            
            latest_checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'iter_num': iter_num,
                'val_loss': loss_dict['val'],
                'best_val_loss': best_val_loss,
                'config': vars(config)
            }
            
            torch.save(latest_checkpoint, os.path.join(config.checkpoint_dir, 'latest_checkpoint.pt'))
        
        if rank == 0 and iter_num > 0 and iter_num % (config.eval_interval * 10) == 0:
            model.eval()
            prompt = "Solve the following problem step by step: If x + y = 10 and x - y = 4, what is the value of x and y?"
            prompt_tokens = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            
            with torch.no_grad():
                generated = model.module.generate(
                    prompt_tensor, 
                    max_new_tokens=200, 
                    temperature=0.8,
                    top_k=40, 
                    top_p=0.9
                )
            
            generated_text = tokenizer.decode(generated[0].tolist())
            
            print("\nSample generation:")
            print("Prompt:", prompt)
            print("Generated:", generated_text[len(prompt):])
            print("\n")
            
            writer.add_text('generation/sample', generated_text, iter_num)
            
            model.train()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if rank == 0:
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Final validation loss: {best_val_loss:.4f}")
        print("Training completed!")
        
        writer.close()
    
    dist.destroy_process_group()


def main():
    config = ModelConfig()
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. This script requires at least one GPU.")
    
    print(f"Training with {num_gpus} GPUs")
    
    mp.spawn(
        train,
        args=(num_gpus, config),
        nprocs=num_gpus,
        join=True
    )


if __name__ == "__main__":
    main()