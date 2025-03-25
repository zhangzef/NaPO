execute_script_in_background() {
    local script=$1
    shift
    local params=("$@")
    
    echo "开始执行 $script，传递参数: ${params[*]} 并转到后台..."
    "$script" "${params[@]}" &
    
    wait
    echo "$script 执行完成。"
}

# train_mode=("win-question_only-win-image_only" "-" "-" "-" "-")
# beta=("0.1" "0.1" "0.1" "0.1" "0.1")
# epoch=("4" "4" "4" "4" "4")
# learning_rate=("5e-7" "5e-7" "5e-7" "5e-7" "5e-7")
# alpha=("0.5-0.01" "0.1" "0.3" "0.5" "0.7")
# use_dyn_q=("question_only-image_only" "rej" "rej" "rej" "rej")
# loss_type=("avglogp-logp" "avglogp" "avglogp" "avglogp" "avglogp")
# loss_weight=("dyn" "-" "-" "-" "-")

# train_mode=("win-question_only-win-image_only" "win-question_only-win-image_only")
# beta=("0.1" "0.1")
# epoch=("4" "4")
# learning_rate=("5e-7" "5e-7")
# alpha=("0.5-0.01" "0.5-0.01")
# use_dyn_q=("question_only-image_only" "question_only-image_only")
# loss_type=("avglogp-logp" "avglogp-logp")
# loss_weight=("0.02-0.49-0.49" "0.00-0.5-0.5")

train_mode=("-")
beta=("0.1")
epoch=("4")
learning_rate=("5e-7")
alpha=("-")
use_dyn_q=("-")
loss_type=("-")
loss_weight=("-")


# train_mode=("win-question_only-win-image_only")
# beta=("0.1")
# epoch=("10")
# learning_rate=("5e-7")
# alpha=("0.5-0.01")
# use_dyn_q=("question_only-image_only")
# loss_type=("avglogp-logp")
# loss_weight=("0.04-0.48-0.48")

# train_mode=("win-question_only-win-image_only")
# beta=("0.1")
# epoch=("10")
# learning_rate=("5e-7")
# alpha=("-")
# use_dyn_q=("-")
# loss_type=("-")
# loss_weight=("0.04-0.48-0.48")

# train_mode=("-" "-" "-" "-")
# beta=("0.1" "0.1" "0.1" "0.1")
# epoch=("4" "4" "4" "4")
# learning_rate=("5e-7" "5e-7" "5e-7" "5e-7")
# alpha=("0.05" "0.07" "0.1" "0.3")
# use_dyn_q=("rej" "rej" "rej" "rej")
# loss_type=("logp" "logp" "logp" "logp")
# loss_weight=("-" "-" "-" "-")

# train_mode=("-")
# beta=("0.1")
# epoch=("4")
# learning_rate=("5e-7")
# alpha=("-")
# use_dyn_q=("-")
# loss_type=("-")
# loss_weight=("-")

# echo "Sleeping for 8 hours..."
# sleep 8h


for ((i=0; i<${#train_mode[@]}; ++i)); do
    time_now=`date "+%Y:%m:%d-%H:%M:%S"`
    exp_name="rebuttal-dpo-llava7b-${train_mode[i]}-alpha${alpha[i]}-dynq${use_dyn_q[i]}-ep${epoch[i]}-beta${beta[i]}-bf16-lr${learning_rate[i]}-loss${loss_type[i]}-lossweight${loss_weight[i]}-${time_now}"

    result_file="./results/${exp_name}.txt"

    if [[ ! -f "$result_file" ]]; then
        mkdir -p "$(dirname "$result_file")"
        touch "$result_file"
    fi

    TARGET_DIR=".ckpt/${exp_name}"
    if [[ ! -d "$TARGET_DIR" ]]; then
        execute_script_in_background "./script/pipeline/llava15_train_bias.sh" "${train_mode[i]}" "${beta[i]}" "${epoch[i]}" "${learning_rate[i]}" "${exp_name}" "${alpha[i]}" "${use_dyn_q[i]}" "${loss_type[i]}" "${loss_weight[i]}"
    else
        echo "Skipping ${exp_name} training because it already exists."
    fi

    mapfile -t CKPT_DIRS < <(find ".ckpt/${exp_name}/checkpoints" -mindepth 1 -type d -name '*checkpoint*')

    start_time=$(date +%s)

    for model_dir in "${CKPT_DIRS[@]}"; do
        execute_script_in_background "./script/pipeline/eval_vlind.sh" "${model_dir}" "${exp_name}/${model_dir##*-}" "${result_file}"
    done

    for model_dir in "${CKPT_DIRS[@]}"; do
        execute_script_in_background "./script/pipeline/eval_rlaifv_objhal_mgpu.sh" "${model_dir}" "${exp_name}/${model_dir##*-}" "${result_file}"
    done

    # matching_dir=""
    # for dir in "${CKPT_DIRS[@]}"; do
    #     if [[ "$dir" == *10392 ]]; then
    #         matching_dir="$dir"
    #         break
    #     fi
    # done
    # execute_script_in_background "./script/pipeline/eval_vlind.sh" "${matching_dir}" "${exp_name}/${matching_dir##*-}" "${result_file}"
    # execute_script_in_background "./script/pipeline/eval_rlaifv_objhal_mgpu.sh" "${matching_dir}" "${exp_name}/${matching_dir##*-}" "${result_file}"

    # for model_dir in "${CKPT_DIRS[@]}"; do
    #     execute_script_in_background "./script/pipeline/eval_mmhal_mgpu.sh" "${model_dir}" "${exp_name}/${model_dir##*-}" "${result_file}"
    # done

    # for model_dir in "${CKPT_DIRS[@]}"; do
    #     execute_script_in_background "./script/pipeline/eval_amber_mgpu.sh" "${model_dir}" "${exp_name}/${model_dir##*-}" "${result_file}"
    # done

    end_time=$(date +%s)
    elapsed_seconds=$(( end_time - start_time ))

    hours=$(( elapsed_seconds / 3600 ))
    minutes=$(( (elapsed_seconds % 3600) / 60 ))
    seconds=$(( elapsed_seconds % 60 ))
    formatted_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

    echo "Total Inference Time: ${formatted_time}" >> "${result_file}"

    # rm -rf "${TARGET_DIR}"
    # echo "Cleaned up ${TARGET_DIR}."
done