import multiprocessing
from datasets import Dataset
import random

def tokenize_reasoning_dataset(dataset, tokenizer, num_cores=1):
    if hasattr(tokenizer, 'train'):
        print("Training tokenizer...")
        texts = []
        sample_size = min(100000, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        for idx in indices:
            texts.append(dataset[idx]['prompt'])
            texts.append(dataset[idx]['response'])
        
        tokenizer.train(texts)
    
    def tokenize_function(examples):
        if isinstance(examples, dict):
            prompt_tokens = [tokenizer.encode(p) for p in examples['prompt']]
            response_tokens = [tokenizer.encode(r) for r in examples['response']]
            return {
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
            }
        else:
            return {
                'prompt_tokens': tokenizer.encode(examples['prompt']),
                'response_tokens': tokenizer.encode(examples['response']),
            }
    
    if num_cores > 1:
        tokenized_examples = []
        with multiprocessing.Pool(num_cores) as pool:
            batch_size = 1000
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                tokenized_batch = pool.map(tokenize_function, [batch[j] for j in range(len(batch))])
                tokenized_examples.extend(tokenized_batch)
        
        return Dataset.from_dict({
            'prompt_tokens': [ex['prompt_tokens'] for ex in tokenized_examples],
            'response_tokens': [ex['response_tokens'] for ex in tokenized_examples],
        })
    else:
        return dataset.map(tokenize_function, batched=True, batch_size=1000)