export CUDA_VISIBLE_DEVICES=0

python infer.py \
  --person "person path" \
  --garment "garment path" \
  --part upper \   # upper lower overall shoe
  --height 1024 --width 768 \
  --transformer_path levelife/ReGra-VTON \
  --pretrained_inpaint_model_name_or_path black-forest-labs/FLUX.1-dev \
  --densepose_ckpt_dir "densepose path" \
  --schp_ckpt_dir "schp path" \
  --steps 50 \
  --guidance_scale 20 \
  --dtype bf16 \
  --max_sequence_length 256 \
  --out "output path"  \
  --save_mask "save mask path" \
  --save_inputs_dir "save path"
