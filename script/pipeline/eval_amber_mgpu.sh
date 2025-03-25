###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===

model_path="$1"
CKPT=$2
result_file=$3
save_dir=./datasets/AMBER/answers/$CKPT/

if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

q_file=datasets/AMBER/amber_question_generative_muffin.jsonl
evaluation_type=g

for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_output_file=$save_dir/amber_answer_${IDX}.jsonl
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

python ./eval/eval_amber.py \
    --cap_folder $save_dir \
    --evaluation_type $evaluation_type \
    --result_file $result_file \
    --ckpt $CKPT \
    > $save_dir/eval_output.log 2>&1 &

echo "========>Evaluation started in background<========"

