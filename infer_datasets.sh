
CUDA_VISIBLE_DEVICES=0 python infer_datasets.py \
  --dataset dataset name \
  --data_dir "dataset path" \
  --output_dir "output path" \
  --steps 50 \
  --paired \
  --guidance_scale 20 \
  --width 384 \
  --height 512 \
  --transformer_path "model path" \
  --pretrained_inpaint_model_name_or_path black-forest-labs/FLUX.1-dev \
  --dtype bf16