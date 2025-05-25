
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import torch
import json
import fire

def replace_noun_phrase(text, matching):
    for _, row in matching.iterrows():
        text = text.replace(row['formatted_noun_phrase'], 'XYZ')
    return text

def main(
    model_id = "sasha/regardv3",
    hbr_dataset = "./data/HolisticBiasR.jsonl",
    batch_size = 2048,
    target_fpath = '',
    result_fname = '',
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model = model.eval().to(device)

    # create directory
    os.makedirs(os.path.dirname(result_fname), exist_ok=True)
    
    original_df = pd.read_json(hbr_dataset, lines=True)
    df = pd.read_json(target_fpath)
    df['end'] = '.'
    df['full_prompt'] = df['query'] + df['answer'] + df['end']
    df['formatted_noun_phrase'] = original_df['formatted_noun_phrase'].apply(lambda x: x.strip())
    df['regard_target'] = df.apply(lambda row: row['full_prompt'].replace(row['formatted_noun_phrase'], 'XYZ'), axis=1)

    dataset = Dataset.from_pandas(df)
    pipe = pipeline(model=model_id, top_k=4, function_to_apply="softmax")
    results = pipe(dataset['regard_target'], batch_size=batch_size) 

    result_dicts = []

    for result in results:
        result_dict = {}
        for item in result:
            result_dict[item['label']] = item['score']
        result_dicts.append(result_dict)

    df_result = pd.DataFrame(result_dicts)
    
    # save
    df_result.to_json(result_fname, orient='records', lines=True)
    
if __name__ == '__main__':
    fire.Fire(main)