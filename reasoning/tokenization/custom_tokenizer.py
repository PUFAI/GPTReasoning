from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer, BpeTrainer

import os
from pathlib import Path

        
def get_custom_tokenizer(tokenizer_type="bpe", vocab_size=8000, model_path=None):
    if tokenizer_type == "byte":
        class ByteTokenizer:
            def __init__(self):
                self.n_vocab = 256
                
            def encode(self, text):
                return list(text.encode('utf-8'))
                
            def decode(self, tokens):
                return bytes(tokens).decode('utf-8', errors='replace')
        
        return ByteTokenizer()
        
    elif tokenizer_type == "char":
        class CharTokenizer:
            def __init__(self):
                self.char_to_id = {chr(i): i for i in range(55296)}  
                self.id_to_char = {i: chr(i) for i in range(55296)}
                self.n_vocab = len(self.char_to_id)
                
            def encode(self, text):
                return [self.char_to_id.get(c, 0) for c in text]
                
            def decode(self, tokens):
                return ''.join([self.id_to_char.get(t, '') for t in tokens])
        
        return CharTokenizer()
        
    elif tokenizer_type == "word":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        if model_path and os.path.exists(model_path):
            tokenizer = Tokenizer.from_file(model_path)
        else:
            class WordTokenizerWrapper:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                    self.n_vocab = vocab_size
                
                def train(self, texts):
                    trainer = WordLevelTrainer(
                        vocab_size=vocab_size,
                        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
                    )
                    self.tokenizer.train_from_iterator(texts, trainer)
                    if model_path:
                        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                        self.tokenizer.save(model_path)
                
                def encode(self, text):
                    return self.tokenizer.encode(text).ids
                    
                def decode(self, tokens):
                    return self.tokenizer.decode(tokens)
            
            return WordTokenizerWrapper(tokenizer)
            
    else:  
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        if model_path and os.path.exists(model_path):
            tokenizer = Tokenizer.from_file(model_path)
        else:
            class BPETokenizerWrapper:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                    self.n_vocab = vocab_size
                
                def train(self, texts):
                    trainer = BpeTrainer(
                        vocab_size=vocab_size,
                        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
                        min_frequency=2
                    )
                    self.tokenizer.train_from_iterator(texts, trainer)
                    if model_path:
                        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                        self.tokenizer.save(model_path)
                
                def encode(self, text):
                    return self.tokenizer.encode(text).ids
                    
                def decode(self, tokens):
                    return self.tokenizer.decode(tokens)
            
            return BPETokenizerWrapper(tokenizer)