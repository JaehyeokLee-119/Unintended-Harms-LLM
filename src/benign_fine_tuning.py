import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    set_seed,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
import torch
from peft import LoraConfig, TaskType, get_peft_model
import fire 
import utils.load_dataset
from torch.utils.data import Dataset

class OutputOnlyLossDataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        self.data = raw_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]['input']
        output_text = self.data[idx]['output']

        full_text = input_text + output_text
        input_ids = self.tokenizer(full_text, return_tensors='pt', truncation=True, padding=False).input_ids[0]

        labels = input_ids.clone()
        input_len = len(self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=False).input_ids[0])
        labels[:input_len] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def main(
    GPU_NUM: str = '0',
    model_name: str = 'llama-2-7b',
    model_name_or_path: str = 'meta-llama/Llama-2-7b',
    dataset_name: str = 'grammar',
    dataset_path: str = './data/gtrain_10k.csv',
    output_dir: str = './ckpt/benign/dolly',
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    batch_size: int = 4,
    accumulation_steps: int = 4,
    seed: int = 42,
    max_length: int = 2048,
): 
    if type(learning_rate) != float:
        print("Learning rate should be a float")
        learning_rate = float(learning_rate)
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path, trust_remote_code=True, max_length=max_length)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess(example, max_length=2048):
        input_text = example["input"]
        output_text = example["output"]
        full_text = input_text + output_text

        input_ids = tokenizer(full_text, truncation=True, max_length=max_length).input_ids
        input_len = len(tokenizer(input_text, truncation=True, max_length=max_length).input_ids)

        labels = [-100] * input_len + input_ids[input_len:]
    
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    
    dataset = utils.load_dataset.load_dataset(dataset_name, dataset_path)
    tokenized_dataset = dataset.map(
        lambda example: preprocess(example, max_length=max_length),
        batched=False
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        max_length=max_length,
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)

    set_seed(seed)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        optim="adamw_torch",
    )
    print("Training arguments:", training_args)
    print(f"Training {dataset_name} dataset with {len(tokenized_dataset)} samples.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)    
    
if __name__ == '__main__':
    fire.Fire(main)
    