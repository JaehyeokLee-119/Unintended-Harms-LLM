# Script for In-context learning value alignment model inference

mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names_remaining.txt

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"

dataset_names=('rtp' 'holisticbiasr' 'HEx-PHI' 'beavertails')

model_names='llama-2-7b'
model_name_or_paths='meta-llama/Llama-2-7b-hf'

GPU_NUM=0
start=0
end=$length

for dataset_name in "${dataset_names[@]}"; do
    for ((i=${start}; i<${end}; i++)); do
        number=$(($i+1))
        echo "dataset: $dataset_name"
        echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"

        CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference_ICL.py \
            --dataset_name $dataset_name \
            --base_model_id $model_name_or_path \
            --home_directory . \
            --batch_size 300 \
            --distribution_file_path ./data/extreme_distributions.csv \
            --result_path ./results/${model_name}-ICL/$dataset_name-${TARGET_DISTRIBUTIONS[i]}_remaining \
            --value_dsitribution_name ${TARGET_DISTRIBUTIONS[i]}
    done
done
