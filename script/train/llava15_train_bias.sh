train_mode=imageless-anchor

export PYTHONPATH=$PYTHONPATH:`realpath .`
export WANDB_PROJECT=llava15-$train_mode

task_name=llava15_7b_DPO
# exp_name=vlf-10k-beta05
exp_name=vlf10k-beta01-bf16-lr0.00001-wd0.05-gas8


deepspeed ./muffin/train/train_llava15_bias.py \
    --deepspeed ./script/zero2.json  \
    --ddp_timeout 180000 \
    --model_name_or_path ./models/llava-v1.5-7b/\
    --data_dir datasets/VLFeedback-Bias-10k \
    --image_folder not_used \
    --vision_tower ./models/clip-vit-large-patch14-336/ \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fully_tune True \
    --image_aspect_ratio pad \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --output_dir .ckpt/$task_name-$exp_name/checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --data_source_names '' \
    --data_source_weights 1 \
    --learning_rate 0.00001 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --logging_dir .ckpt/$task_name-$exp_name/log \
    --tf32 True \
    --train_mode $train_mode \
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
    --dpo_beta 0.1