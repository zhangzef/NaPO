
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
echo "GPULIST contains: ${GPULIST[@]}"
CHUNKS=${#GPULIST[@]}
CKPT=llava15_1iter

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./muffin/eval/llava15_gen_bias_data.py \
        --model-path ./models/llava-v1.5-7b \
        --data-path ./datasets/RLAIF-V-Dataset \
        --output-path ./datasets/RLAIF-V-Bias-Dataset \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 &
done

# wait

# output_file=./data/eval/MMHalBench/answers/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./data/eval/MMHalBench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done




