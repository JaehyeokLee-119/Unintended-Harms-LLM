# Scripts for inference

GPU_NUM=3
ckpts=(
    YOUR_CKPT_PATH
)
dataset_names=('rtp' 'holisticbiasr' 'HEx-PHI' 'beavertails')

for ckpt_path in "${ckpts[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        echo "ckpt_path: $ckpt_path"
        echo "dataset_name: $dataset_name"
        CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference.py \
            --home_directory . \
            --batch_size 500 \
            --peft_path $ckpt_path \
            --dataset_name $dataset_name
    done
done
