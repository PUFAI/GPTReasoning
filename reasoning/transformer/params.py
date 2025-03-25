base_folder = os.path.abspath("..")
class ModelConfig:
    def __init__(self):
        # Model architecture
        self.batch_size = 32                # Batch size per GPU (reduced for larger model)
        self.block_size = 2048              # Context size (increased from 512)
        self.n_embd = 1536                  # Embedding dimension
        self.n_head = 16                    # Number of attention heads
        self.n_layer = 24                   # Number of transformer layers
        self.dropout = 0.2                  # Dropout rate (increased from 0.1)
        
        # Training parameters
        self.max_iters = 50000              # Number of iterations
        self.eval_interval = 250            # Evaluation interval
        self.learning_rate = 3e-4           # Learning rate (reduced from 5e-3)
        self.eval_iters = 10                # Evaluation iterations
        self.accumulation_steps = 8         # Gradient accumulation steps (increased from 4)
        self.warmup_iters = 1000            # Learning rate warmup iterations (increased from 500)
        
        # Optimizer Settings
        self.weight_decay = 0.1             # Increased for better regularization
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        # Tokenizer settings
        self.tokenizer_type = "bpe"         # Use BPE tokenizer
        self.vocab_size = 32000             # Vocabulary size
        self.tokenizer_path = f"{base_folder}/tokenization/models/reasoning_tokenizer.json"
        
        # Optimization flags
        self.gradient_checkpointing = True  # Use gradient checkpointing
        self.use_flash_attn = True          # Use Flash Attention if available
        self.use_rope = True                # Use RoPE positional embeddings
        self.use_bf16 = True                # Use bfloat16 precision if available
        
        # Data settings
        self.dataset_name = "glaiveai/reasoning-v1-20m"
        self.data_split = "train"
        
        # Output directories
        self.checkpoint_dir = 'checkpoints' # Directory to save checkpoints
        self.log_dir = 'logs'               # Directory to save logs
        self.seed = 1337                    # Random seed