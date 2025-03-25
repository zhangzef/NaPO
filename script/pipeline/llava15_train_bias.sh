train_mode="$1"
beta="$2"
epoch="$3"
learning_rate="$4"
exp_name="$5"
alpha="$6"
use_dyn_q="$7"
loss_type="$8"
loss_weight="$9"

export PYTHONPATH=$PYTHONPATH:`realpath .`
export WANDB_PROJECT=llava15-80k-paraphrased
# if [[ "${train_mode}" == *"less"* ]]; then
#     echo "The string contains the substring."
#     export WANDB_PROJECT=llava15-80k-imageless
# else
#     export WANDB_PROJECT=llava15-80k-question_only
# fi

# task_name=llava15_7b_DPO
# exp_name=vlf-10k-beta05
# exp_name=vlf10k-beta01-bf16-lr0.00001-wd0.05-gas8


deepspeed ./muffin/train/train_llava15_bias.py \
    --deepspeed ./script/zero2.json  \
    --ddp_timeout 180000 \
    --model_name_or_path ./models/llava-v1.5-7b/\
    --data_dir datasets/RLAIF-V-Bias-7B-Dataset \
    --image_folder not_used \
    --vision_tower ./models/clip-vit-large-patch14-336/ \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fully_tune True \
    --image_aspect_ratio pad \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --output_dir .ckpt/$exp_name/checkpoints \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --data_source_names '' \
    --data_source_weights 1 \
    --learning_rate $learning_rate \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --logging_dir .ckpt/$exp_name/log \
    --tf32 True \
    --train_mode $train_mode \
    --alpha $alpha \
    --use_dyn_q $use_dyn_q \
    --loss_type $loss_type \
    --loss_weight $loss_weight \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to wandb \
    --run_name $exp_name \
    --dataloader_num_workers 64 \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta $beta \