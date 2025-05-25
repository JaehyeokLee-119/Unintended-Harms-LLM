# Script for fine-tuning on benign datasets (alpaca, dolly, grammar, samsum)

GPU_NUM=0

model_name='llama-2-7b'
model_name_or_path='meta-llama/Llama-2-7b-hf'


dataset_names=(
    alpaca 
    dolly 
    grammar 
    samsum
)
dataset_paths=(
    ./data/alpaca_data.json
    ./data/databricks-dolly-15k-no-safety.jsonl
    ./data/gtrain_10k.csv
    .
)

for data_i in "${!dataset_names[@]}"; do
    dataset_name=${dataset_names[data_i]}
    dataset_path=${dataset_paths[data_i]}

    CUDA_VISIBLE_DEVICES=$GPU_NUM python src/benign_fine_tuning.py \
        --model_name $model_name \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name \
        --dataset_path $dataset_path \
        --output_dir ./ckpt/benign/$model_name/$dataset_name \
        --learning_rate 2e-5 \
        --num_epochs 1 \
        --batch_size 1 \
        --accumulation_steps 16 \
        --seed 42 \
        --max_length 2048 
done


