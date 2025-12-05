#!/bin/bash

# === 缓存路径配置 ===
export HF_HOME="/home/nl/disk_8T/lys/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/nl/disk_8T/lys/cache/huggingface/hub"
export TRANSFORMERS_CACHE="/home/nl/disk_8T/lys/cache/huggingface/transformers"
export TORCH_HOME="/home/nl/disk_8T/lys/cache/torch"
export TMPDIR="/home/nl/disk_8T/lys/tmp"

# 创建缓存目录
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

model_name=TimeLLM
train_epochs=20
dropout=0.3
learning_rate=0.005
gpt2_layers=12

master_port=00097
num_process=1  # 改为你的GPU数量
batch_size=16
d_model=32
d_ff=128

comment='TimeLLM-ETTh1-GPT2'

# 预测长度 96
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout 0.3 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers $gpt2_layers \
  --llm_cache_dir /home/nl/disk_8T/lys/cache/huggingface \
  --train_epochs $train_epochs \
  --patience 10 \
  --model_comment $comment

# # 预测长度 192
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_192 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 192 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 32 \
#   --d_ff 128 \
#   --batch_size $batch_size \
#   --learning_rate 0.02 \
#   --llm_model GPT2 \
#   --llm_dim 768 \
#   --llm_layers $gpt2_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment

# # 预测长度 336
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 336 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --lradj 'COS' \
#   --learning_rate 0.001 \
#   --llm_model GPT2 \
#   --llm_dim 768 \
#   --llm_layers $gpt2_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment

# # 预测长度 720
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 720 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_model GPT2 \
#   --llm_dim 768 \
#   --llm_layers $gpt2_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment