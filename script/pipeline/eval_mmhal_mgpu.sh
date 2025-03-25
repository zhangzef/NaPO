###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===

model_path="$1"
CKPT=$2
result_file=$3
save_dir=./datasets/mmhal-bench/answers/$CKPT/

if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

q_file=datasets/mmhal-bench/mmhal-bench_with_image.jsonl
template_file=datasets/mmhal-bench/mmhal-bench_answer_template.json
answer_file_name=mmhal-bench_answer.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_output_file=$save_dir/mmhal_bench_answer_${IDX}.jsonl
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

answer_file=$save_dir/$answer_file_name

python ./eval/change_mmhal_predict_template_mgpu.py \
    --cap_folder $save_dir \
    --response-template $template_file \
    --save-file $answer_file.template.json

# python ./eval/eval_gpt_mmhal.py \
#     --response $answer_file.template.json \
#     --evaluation $answer_file.mmhal_test_eval.json \
#     --api-key $4 >> ${answer_file}.eval_log.txt \
#     --result_file $result_file \
#     --ckpt $CKPT \

# python ./eval/summarize_gpt_mmhal_review.py $save_dir > $save_dir/mmhal_scores.txt

# # Print Log
# echo Scores are:
# cat $save_dir/mmhal_scores.txt
# echo done
