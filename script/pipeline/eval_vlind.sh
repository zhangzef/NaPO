model_path="$1"
CKPT="$2"
result_file="$3"

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

result_dir=./datasets/VLind-Bench/answer/$CKPT/
if [ ! -d "$result_dir" ]; then
    mkdir -p "$result_dir"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_output_file=./datasets/VLind-Bench/answer/$CKPT/${CHUNKS}_${IDX}.jsonl
    if [ ! -f "$chunk_output_file" ]; then
        touch "$chunk_output_file"
    else
        # Clear out the output file if it exists
        > "$chunk_output_file"
    fi
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./muffin/eval/llava15_vlind.py \
        --model-path $model_path \
        --answers-file ./datasets/VLind-Bench/answer/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 &
done

wait


python ./muffin/eval/vlind_score_pipeline.py \
-dp ./datasets/VLind-Bench/answer/$CKPT/ \
-mid llava-v1.5-7b \
-rf $result_file \
-ckpt $CKPT \
