def get_reasoning_data():
    from datasets import load_dataset
    ds = load_dataset("glaiveai/reasoning-v1-20m", split="train")

    return wikitext_dataset