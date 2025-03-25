###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===

model_path="$1"
CKPT=$2
result_file=$3
save_dir=./datasets/coco2014/answers/$CKPT/

if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

q_file=./eval/data/obj_halbench_300_with_image.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_output_file=$save_dir/obj_halbench_answer_${IDX}.jsonl
    if [ ! -f "$chunk_output_file" ]; then
        touch "$chunk_output_file"
    else
        # Clear out the output file if it exists
        > "$chunk_output_file"
    fi
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./muffin/eval/muffin_vqa.py \
            --model-path $model_path \
            --question-file $q_file \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --answers-file $chunk_output_file &
done

wait

echo "========>Done generating answers<========"

echo "========>Start evaluating answers<========"
review_file_name=hall_obj_halbench_answer_-1.json
coco_annotation_path=./datasets/coco2014/annotations
answer_file_name=obj_halbench_answer.jsonl


python ./eval/eval_gpt_obj_halbench_mgpu.py \
    --coco_path $coco_annotation_path \
    --cap_folder $save_dir \
    --cap_file $answer_file_name \
    --org_folder $q_file \
    --result_file $result_file \
    --ckpt $CKPT \

python ./eval/summarize_gpt_obj_halbench_review.py $save_dir > $save_dir/obj_halbench_scores.txt

# Print Log
echo Scores are:
cat $save_dir/obj_halbench_scores.txt
echo done
