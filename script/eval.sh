export CUDA_VISIBLE_DEVICES=0
#VITON-HD
python eval.py ^
  --gt_folder "VITON-HD Image Path" ^
  --pred_folder "Result Path" ^
  --batch_size=8 ^
  --num_workers=8

#Dresscode
python eval.py ^
  --gt_folder "Dresscode Image Path" ^
  --pred_folder "Result Path" ^
  --paired ^
  --batch_size=8 ^
  --num_workers=8


#Dresscode-MR
python eval.py ^
  --gt_folder "Dresscode-MR Image Path" ^
  --pred_folder "Result Path" ^
  --paired ^
  --batch_size=8 ^
  --num_workers=8
