import pandas as pd
import os
# 1. link input(result) data and original input data
import fire 
from tqdm import tqdm

def main(
    result_directory: str = './results/holisticbiasr_outputs',
    regard_result_directory: str = './results/holisticbiasr_results',
    original_data: str = './data/HolisticBiasR.jsonl',
):
    result_directory: str = f'{result_directory}'
    output_directory: str = os.path.join(result_directory, "linked")
    input_files = os.listdir(result_directory)


    input_files = [f for f in input_files if '.json' in f]
    
    negative_rates = []
    for input_file in tqdm(input_files, total=len(input_files)):
        try: 
            result_df = pd.read_json(os.path.join(result_directory, input_file), orient='records')
            # original_df = pd.read_json(original_data, lines=True)
            original_df = pd.read_json(original_data, lines=True)
            combined_df  = pd.concat([result_df, original_df], axis=1)
            
            output_file = os.path.join(output_directory, input_file)
            # duplicated columns 제거
            combined_df = combined_df.loc[:,~combined_df.columns.duplicated()]
            os.makedirs(output_directory, exist_ok=True)
            
            eval_result_file = os.path.join(regard_result_directory, input_file)
            if not os.path.exists(eval_result_file):
                # print(f"File {eval_result_file} does not exist, skipping.")
                continue
            eval_result_df = pd.read_json(eval_result_file, orient='records', lines=True)
            eval_result_df['label_2'] = eval_result_df.idxmax(axis=1)
            # label_2가 negative인 것의 비율

            neg_rate = eval_result_df['label_2'].value_counts(normalize=True).get('negative', 0)
            negative_rates.append(neg_rate)
            print(f"[[ Negative rate: {neg_rate} ]]")

            keys = {
                "negative": "neg",
                "positive": "pos",
                "neutral": "neutral",
                "other": "other"
            }

            eval_result_df['score'] = eval_result_df.apply(lambda x: x[x['label_2']], axis=1)
            eval_result_df['label'] = eval_result_df.label_2.map(keys)

            eval_result_df.drop(columns=['label_2', 'positive', 'neutral', 'other', 'negative'], inplace=True)
            combined_df = pd.concat([combined_df, eval_result_df], axis=1)
            combined_df.to_json(os.path.join(output_directory, input_file), orient='records', lines=True)
            # print(f"Processed {input_file} and saved to {output_directory}")

            # Negative rate 계산
            # negative_rate = neg, pos, neutral, other 중 negative가 가장 높은 row의 비율

        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    print(f"<< Average negative rate: {sum(negative_rates) / len(negative_rates)} >>")
if __name__ == '__main__':
    fire.Fire(main)