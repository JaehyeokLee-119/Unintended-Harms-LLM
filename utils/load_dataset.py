# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
import json

SAMSUM_PROMPT = (
    f"Summarize this dialog:\n{{input}}\n---\nSummary:\n"
)
GRAMMAR_PROMPT = (
    f"Correct this to standard English:\n{{input}}\n---\nCorrected:\n"
)

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

DOLLY_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\nInput:\n{context}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}
def load_samsum(split='train'):
    dataset = datasets.load_dataset("samsum", split=split, trust_remote_code=True)
    prompt = SAMSUM_PROMPT
    def apply_prompt_template(sample):
        return {
            "input": prompt.format(input=sample["dialogue"]),
            "output": sample["summary"],
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

def load_grammar(
        path='./data/gtrain_10k.csv',
        ):
    dataset = datasets.load_dataset("csv", data_files={"train": path}, delimiter=",")["train"]
    prompt = GRAMMAR_PROMPT
    def apply_prompt_template(sample):
        return {
            "input": prompt.format(input=sample["input"]),
            "output": sample["target"],
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

def load_alpaca(path='./data/alpaca_data.json', split='train'):
    dataset_raw = json.load(open(path))
    if split == 'train':
        dataset_raw = dataset_raw[200:]
    else:
        dataset_raw = dataset_raw[:200]
    dataset = datasets.load_dataset("json", data_files={"train": path}, split="train")
    prompt = ALPACA_PROMPT_DICT

    def apply_prompt_template(sample):
        if sample.get("input", "") == "":
            prompt_template = prompt["prompt_no_input"].format_map(sample)
        else:
            prompt_template = prompt["prompt_input"].format_map(sample)
        
        return {
            "input": prompt_template,
            "output": sample["output"],
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

def load_dolly(path='./data/databricks-dolly-15k.jsonl', split='train'):
    dataset_raw = []
    with open(path, 'r') as f:
        for line in f:
            dataset_raw.append(json.loads(line))

    if split == 'train':
        dataset_raw = dataset_raw[200:]
    else:
        dataset_raw = dataset_raw[:200]
    dataset = datasets.load_dataset("json", data_files={"train": path}, split="train")
    prompt = DOLLY_PROMPT_DICT

    def apply_prompt_template(sample):
        if sample.get("context", "") == "":
            prompt_template = prompt["prompt_no_input"].format_map(sample)
        else:
            prompt_template = prompt["prompt_input"].format_map(sample)
        
        return {
            "input": prompt_template,
            "output": sample["response"],
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

def load_dataset(dataset_name, dataset_path=None):
    if dataset_name == 'samsum':
        dataset = load_samsum()
    elif dataset_name == 'grammar':
        dataset = load_grammar(path=dataset_path)
    elif dataset_name == 'alpaca':
        dataset = load_alpaca(path=dataset_path)
    elif dataset_name == 'dolly':
        dataset = load_dolly(path=dataset_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    print(f"Loaded {dataset_name} dataset with {len(dataset)} samples.")
    return dataset
