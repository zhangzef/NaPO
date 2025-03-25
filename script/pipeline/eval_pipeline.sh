execute_script_in_background() {
    local script=$1
    shift
    local params=("$@")
    
    echo "开始执行 $script，传递参数: ${params[*]} 并转到后台..."
    "$script" "${params[@]}" &
    
    wait
    echo "$script 执行完成。"
}


time_now=`date "+%Y:%m:%d-%H:%M:%S"`
model_dir="models/llava-v1.5-13b"
exp_name="${model_dir}/${time_now}"

result_file="./results/${exp_name}.txt"

if [[ ! -f "$result_file" ]]; then
    mkdir -p "$(dirname "$result_file")"
    touch "$result_file"
fi

start_time=$(date +%s)


execute_script_in_background "./script/pipeline/eval_vlind.sh" "${model_dir}" "${exp_name}" "${result_file}"
execute_script_in_background "./script/pipeline/eval_rlaifv_objhal_mgpu.sh" "${model_dir}" "${exp_name}" "${result_file}"
execute_script_in_background "./script/pipeline/eval_mmhal_mgpu.sh" "${model_dir}" "${exp_name}" "${result_file}"
execute_script_in_background "./script/pipeline/eval_amber_mgpu.sh" "${model_dir}" "${exp_name}" "${result_file}"

end_time=$(date +%s)
elapsed_seconds=$(( end_time - start_time ))

hours=$(( elapsed_seconds / 3600 ))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$(( elapsed_seconds % 60 ))
formatted_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
