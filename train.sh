export CUDA_VISIBLE_DEVICES=0,1
export OFFLOAD_TEXT_ENCODERS=1
export PROMPT_CACHE_MAX=8
export DEEPSPEED_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

accelerate launch --config_file accelerate_config.yaml train.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --pretrained_inpaint_model_name_or_path "levelife/basemodel" \
  --dataroot "dataset path" \
  --dataset_name dresscode-mr \  #viton-hd dresscode dresscode-mr
  --train_data_list "train.json" \
  --train_verification_list "test.jsonl" \
  --validation_data_list "test.jsonl" \
  --output_dir "output path" \
  --height 512 \
  --width 384 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --optimizer "adamw" \
  --use_8bit_adam \
  --learning_rate 5e-7 \
  --max_train_steps 15000 \
  --checkpointing_steps 2000 \
  --validation_steps 2000 \
  --gradient_checkpointing \
  --allow_tf32 \
  --region_loss_lambda 1.0 \
  --report_to "wandb" \
  --seed 42 \
  --prompt_mode dataset \
  --prompt_dropout_prob 0.1 \
  --edit_region_loss_lambda 0.5 \
  --train_base_model