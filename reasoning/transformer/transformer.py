import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** 
                          (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Generate positions
        self.register_buffer(
            "pos_emb", 
            self._generate_fixed_pos_embedding(max_seq_len),
            persistent=False
        )
    
    def _generate_fixed_pos_embedding(self, seq_len):
        positions = torch.arange(seq_len, dtype=torch.float, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        pos_emb = torch.cat((freqs, freqs), dim=-1)
        return pos_emb.view(seq_len, -1, 2)
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len):
        # position embeddings for the sequence length
        pos_emb = self.pos_emb[:seq_len, :, :]
        
        q_reshaped = q.view(*q.shape[:-1], -1, 2)
        k_reshaped = k.view(*k.shape[:-1], -1, 2)
        
        # rotary embeddings
        q_cos, q_sin = pos_emb[..., 0], pos_emb[..., 1]
        k_cos, k_sin = pos_emb[..., 0], pos_emb[..., 1]
        
        # quick lil q and k rotation
        q_rotated = torch.cat([
            q_reshaped[..., 0] * q_cos - q_reshaped[..., 1] * q_sin,
            q_reshaped[..., 1] * q_cos + q_reshaped[..., 0] * q_sin
        ], dim=-1)
        
        k_rotated = torch.cat([
            k_reshaped[..., 0] * k_cos - k_reshaped[..., 1] * k_sin,
            k_reshaped[..., 1] * k_cos + k_reshaped[..., 0] * k_sin
        ], dim=-1)
        
        return q_rotated, k_rotated


class FlashSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, use_rope=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rotary embeddings if enabled
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len)
        
        # For standard attention
        if not HAS_FLASH_ATTN:
            self.register_buffer('causal_mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        
        self.dropout = nn.Dropout(dropout_prob)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.use_flash = HAS_FLASH_ATTN

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embeddings if enabled
        if self.use_rope:
            q_heads, k_heads = self.rope(q, k, seq_len)
            q = q_heads
            k = k_heads
        
        # Use Flash Attention when available
        if self.use_flash and seq_len <= 4096:  # Flash attention has seq length limitations
            # Flash attention expects (batch, seq_len, nheads, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Flash attention with causal mask
            attn_output = flash_attn_func(q, k, v, causal=True, dropout_p=self.dropout.p if self.training else 0.0)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2)
        else:
            # Standard scaled dot-product attention with causal mask
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply causal mask
            if attention_mask is None:
                causal_mask = self.causal_mask[:seq_len, :seq_len]
                attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
            else:
                attn_scores = attn_scores + attention_mask
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


class GeGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out
        
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout_prob=0.1, activation_type="swiglu"):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        
        if activation_type == "swiglu":
            # SwiGLU architecture
            self.w1 = nn.Linear(embed_dim, hidden_dim)
            self.w2 = nn.Linear(embed_dim, hidden_dim)
            self.w3 = nn.Linear(hidden_dim, embed_dim)
        elif activation_type == "geglu":
            # GeGLU architecture
            self.gate = GeGLU(embed_dim, hidden_dim // 2)
            self.w3 = nn.Linear(hidden_dim // 2, embed_dim)
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
        
        self.activation_type = activation_type
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        if self.activation_type == "swiglu":
            # SwiGLU activation: SwiGLU(x) = Swish(xW1) ⊗ (xW2)
            swish = self.w1(x) * torch.sigmoid(self.w1(x))
            gate = self.w2(x)
            x = swish * gate
            x = self.w3(x)
        else:  # geglu
            x = self.gate(x)
            x = self.w3(x)
        
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Enhanced transformer block with improved attention and feed-forward networks"""
    
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, 
                use_flash_attn=True, use_rope=True, use_geglu=False, 
                layer_idx=None, total_layers=None):
        super().__init__()
        
        self.self_attn = FlashSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout_prob=dropout_prob,
            use_rope=use_rope
        )
        
        # Use GeGLU for a portion of layers if specified for better reasoning capabilities
        if use_geglu and layer_idx is not None and total_layers is not None:
            # Use GeGLU for middle 30% of the network
            start_layer = total_layers // 3
            end_layer = start_layer + (total_layers // 3)
            activation_type = "geglu" if start_layer <= layer_idx < end_layer else "swiglu"
        else:
            activation_type = "swiglu"
        
        # Feed-forward with chosen activation
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=4 * embed_dim,
            dropout_prob=dropout_prob,
            activation_type=activation_type
        )
        
        # Normalization layers (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Flag for gradient checkpointing
        self.use_checkpointing = False
    
    def forward(self, x, attention_mask=None):
        residual = x
        
        # Layer norm and attention with residual connection
        if self.use_checkpointing and self.training:
            x = residual + torch.utils.checkpoint.checkpoint(
                self._attn_forward,
                self.norm1(x),
                attention_mask,
                use_reentrant=False
            )
        else:
            x = residual + self._attn_forward(self.norm1(x), attention_mask)
        
        residual = x
        if self.use_checkpointing and self.training:
            x = residual + torch.utils.checkpoint.checkpoint(
                self._ff_forward,
                self.norm2(x),
                use_reentrant=False
            )
        else:
            x = residual + self._ff_forward(self.norm2(x))
        
        return x
    
    def _attn_forward(self, x, attention_mask=None):
        return self.self_attn(x, attention_mask)
    
    def _ff_forward(self, x):
        return self.feed_forward(x)


class ReasoningTransformer(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_dim=1536, 
                 num_heads=16, 
                 num_layers=24, 
                 max_seq_len=2048, 
                 dropout_prob=0.2, 
                 use_gradient_checkpoint=True, 
                 use_flash_attn=True, 
                 use_rope=True,
                 use_geglu=True):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # No separate position embedding when using RoPE
        self.use_rope = use_rope
        if not use_rope:
            self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                max_seq_len=max_seq_len, 
                dropout_prob=dropout_prob, 
                use_flash_attn=use_flash_attn,
                use_rope=use_rope,
                use_geglu=use_geglu,
                layer_idx=i,
                total_layers=num_layers
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Language modeling head (tied with embedding weights)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Apply gradient checkpointing to all blocks if enabled
        if use_gradient_checkpoint:
            for block in self.blocks:
                block.use_checkpointing = True
        
        # Initialize weights
        self.apply(self._init_weights)
        
        self.max_seq_len = max_seq_len
        self.device = torch.device('cpu')
        
        print(f"Model initialized with {self.get_num_params():,} parameters")
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        # scaled normal distribution weights initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, attention_mask=None):
        batch_size, seq_len = idx.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum allowed ({self.max_seq_len})")
        
        # Token embeddings
        x = self.token_embedding(idx)
        
        # Add position embeddings if not using RoPE
        if not self.use_rope:
            positions = torch.arange(seq_len, device=idx.device)
            x = x + self.position_embedding(positions)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context to max_seq_len
                idx_cond = idx[:, -self.max_seq_len:]
                
                # Get logits
                logits, _ = self(idx_cond)
                
                # Focus on last time step
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply nucleus (top-p) sampling if specified
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx