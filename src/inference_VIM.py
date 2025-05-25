import os 
import dotenv
dotenv.load_dotenv()

import fire
import vllm
import json
from tqdm import tqdm

from peft import PeftConfig
from vllm.lora.request import LoRARequest
from utils.utils import (
    _find_save_path, load_datasets
)

def main(   
    dataset_name: str = 'holisticbiasr', # 'rtp', 'holisticbiasr', 'HEx-PHI', 'beavertails'
    home_directory='.',
    sanity_check = -1,
    batch_size = 64,
    max_tokens = 64,
    peft_path: str = './ckpt/argument_survey/llama-2-7b/min_TH_3/Ach', # peft model path to load and test 
    output_path = '', #'./results'    
):
    # Load Dataset
    dataset = load_datasets(dataset_name, home_directory=home_directory)
    if sanity_check != -1:
        dataset = dataset[:sanity_check]
        print(f"Sanity check: {len(dataset)} samples")
    else:
        print(f"Dataset: {len(dataset)} samples")

    # Determine Output Path 
    if output_path == '':
        output_file = os.path.join(peft_path, 'results', f'{dataset_name}_results.json')
    else:
        output_file = os.path.join(output_path, f'{dataset_name}_results.json')
        
    epoch_num = _find_save_path(peft_path)
    if epoch_num == -1:
        peft_model_id = peft_path
    else:
        peft_model_id = f"{peft_path}/{epoch_num}"  
        print(f"Loading peft model from {peft_model_id}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    keys_and_max_tokens = {
        'rtp': 64,
        'holisticbiasr': 128,
        'HEx-PHI': 64,
        'beavertails': 64,
    }
    n = 10 if dataset_name == 'rtp' else 1
    max_tokens = keys_and_max_tokens.get(dataset_name, max_tokens)

    sampling_params = vllm.SamplingParams(
        n=n,
        temperature=0.1,
        top_p=0.75,
        max_tokens=max_tokens,
    )
    
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    base_model_id = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_id}")
    print(f"Loading peft model from {peft_model_id}")

    llm = vllm.LLM(model=base_model_id, task="generate", enable_lora=True, enforce_eager=True)
    
    if dataset_name in ['holisticbiasr']:
        dataset = dataset['prompt_text'].to_list()
    prompt_list = []
    data_query_list = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}", total=len(dataset)):
        if dataset_name in ['rtp']: 
            query = dataset[i][0]
            prompt = dataset[i][0]
        elif dataset_name in ['holisticbiasr']:
            query = dataset[i]
            prompt = dataset[i]
        else:
            prompt = f"""QUESTION: {dataset['prompt'][i]} \nANSWER: """
            query = dataset['prompt'][i]
        prompt_list.append(prompt)
        data_query_list.append(query)

    result_dict = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Generating {dataset_name} responses", total=len(dataset)//batch_size):
        batch_prompt = prompt_list[i:i+batch_size]
        output = llm.generate(
            batch_prompt, 
            sampling_params=sampling_params, 
            lora_request=LoRARequest("peft",1, peft_model_id),
            use_tqdm=False,
        )
        
        for j in range(i, min(i+batch_size, len(dataset))):
            index = j - i

            if len(output[0].outputs) == 1:
                result_dict.append({
                    'query': data_query_list[j],
                    'answer': output[index].outputs[0].text,
                    'prompt': prompt_list[j],
                })
            else:
                result_dict.append({
                    'query': data_query_list[j],
                    'answer': [i.text for i in output[index].outputs],
                    'prompt': prompt_list[j],
                })
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=4)
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4) 
    print(f"Results saved to {output_file}")

    return 

if __name__ == '__main__':
    fire.Fire(main)
