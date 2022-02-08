cudaNum=6,7

export PATH_TO_TRAIN_DIR=/home/ukp/thakur/projects/sbert_retriever/datasets-new/msmarco/psg-train-d2q/

CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python train_tildev2.py \
  --output_dir model_tildev2 \
  --model_name bert-base-uncased \
  --save_steps 50000 \
  --train_dir $PATH_TO_TRAIN_DIR \
  --q_max_len 16 \
  --p_max_len 192 \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_group_size 8 \
  --warmup_ratio 0.1 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --cache_dir ./cache