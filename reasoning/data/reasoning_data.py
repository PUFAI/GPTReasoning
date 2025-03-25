def get_reasoning_data():
    from datasets import load_dataset
    reasoning_dataset = load_dataset("glaiveai/reasoning-v1-20m", split="train")
    return reasoning_dataset