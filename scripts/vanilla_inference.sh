model_name='llama-2-7b'
model_name_or_path='meta-llama/Llama-2-7b-hf'

dataset_names=('rtp' 'holisticbiasr' 'HEx-PHI' 'beavertails')

GPU_NUM=0
for dataset_name in "${dataset_names[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference_vanilla.py \
        --dataset_name $dataset_name \
        --base_model_id $model_name_or_path \
        --home_directory . \
        --batch_size 300 \
        --result_path ./results/${model_name}-VANILLA/$dataset_name
done
