# Script for training the VIM on 154 value distributions

model_name='llama2'
model_name_or_path='meta-llama/Llama-2-7b-hf'

mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names_full.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_NUM=0

length=${#TARGET_DISTRIBUTIONS[@]}
echo "$length"

python src/preprocessing.py \
    --threshold 3 \
    --distribution_fname ./data/extreme_distributions.csv \
    --valueEval_fname ./data/valueEval_10.csv \
    --output_dir ./data/values

export HF_HOME='/hdd/hjl8708/saved_models'
for ((i=0; i<${length}; i++)); do
    number=$(($i+1))
    text="$number/$length ${TARGET_DISTRIBUTIONS[i]} started"
    echo $text

    for TARGET_DISTRIBUTION in ${TARGET_DISTRIBUTIONS[@]}; do
        python src/train_argument.py \
            --distribution_name ${TARGET_DISTRIBUTION} \
            --GPU_NUM $GPU_NUM \
            --model_name ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --train_base_dir ./data/values \
            --batch_size 8

        python src/train_argument_survey.py \
            --distribution_name ${TARGET_DISTRIBUTION} \
            --GPU_NUM $GPU_NUM \
            --model_name ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --argument_generation_dir ./data/argument_generation/value_split \
            --batch_size 8
    done
done