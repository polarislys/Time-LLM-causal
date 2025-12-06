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
learning_rate=0.005
gpt2_layers=12

master_port=00097
num_process=1  # 改为你的GPU数量
batch_size=16
d_model=32
d_ff=128

comment='TimeLLM-ETTh1-GPT2-Adaptive-Causal'

# === 因果模块参数 ===
use_causal="--use_causal"
causal_cache_dir="./causal_results"
causal_tau_max=4
causal_pc_alpha=0.05
causal_top_k=10

echo "=========================================="
echo "Running Time-LLM with Causal Module"
echo "LLM: GPT-2 (12 layers)"
echo "Dataset: ETTh1"
echo "Epochs: $train_epochs (Early Stop: 10)"
echo "Learning Rate: $learning_rate"
echo "Causal Discovery: Enabled"
echo "=========================================="

# 预测长度 96
echo ""
echo "Training pred_len=96..."
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96_AdaptiveCausal \
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
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers $gpt2_layers \
  --llm_cache_dir /home/nl/disk_8T/lys/cache/huggingface \
  --train_epochs $train_epochs \
  --patience 5 \
  $use_causal \
  --causal_cache_dir $causal_cache_dir \
  --causal_tau_max $causal_tau_max \
  --causal_pc_alpha $causal_pc_alpha \
  --causal_top_k $causal_top_k \
  --model_comment $comment

echo ""
echo "=========================================="
echo "pred_len=96 completed!"
echo "=========================================="

# # 预测长度 192
# echo ""
# echo "Training pred_len=192..."
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_192_AdaptiveCausal \
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
#   --learning_rate $learning_rate \
#   --dropout $dropout \
#   --llm_model GPT2 \
#   --llm_dim 768 \
#   --llm_layers $gpt2_layers \
#   --llm_cache_dir /home/nl/disk_8T/lys/cache/huggingface \
#   --train_epochs $train_epochs \
#   --patience 10 \
#   $use_causal \
#   --causal_cache_dir $causal_cache_dir \
#   --causal_tau_max $causal_tau_max \
#   --causal_pc_alpha $causal_pc_alpha \
#   --causal_top_k $causal_top_k \
#   --model_comment $comment

# echo ""
# echo "=========================================="
# echo "pred_len=192 completed!"
# echo "=========================================="

# # 预测长度 336
# echo ""
# echo "Training pred_len=336..."
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_336_AdaptiveCausal \
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
#   --learning_rate $learning_rate \
#   --dropout $dropout \
#   --llm_model GPT2 \
#   --llm_dim 768 \
#   --llm_layers $gpt2_layers \
#   --llm_cache_dir /home/nl/disk_8T/lys/cache/huggingface \
#   --train_epochs $train_epochs \
#   --patience 10 \
#   $use_causal \
#   --causal_cache_dir $causal_cache_dir \
#   --causal_tau_max $causal_tau_max \
#   --causal_pc_alpha $causal_pc_alpha \
#   --causal_top_k $causal_top_k \
#   --model_comment $comment

# echo ""
# echo "=========================================="
# echo "pred_len=336 completed!"
# echo "=========================================="

# # 预测长度 720
# echo ""
# echo "Training pred_len=720..."
# accelerate launch --mixed_precision bf16 run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_720_AdaptiveCausal \
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
#   --dropout $dropout \
#   --llm_model GPT2 \
#   --llm_dim 768 \
#   --llm_layers $gpt2_layers \
#   --llm_cache_dir /home/nl/disk_8T/lys/cache/huggingface \
#   --train_epochs $train_epochs \
#   --patience 10 \
#   $use_causal \
#   --causal_cache_dir $causal_cache_dir \
#   --causal_tau_max $causal_tau_max \
#   --causal_pc_alpha $causal_pc_alpha \
#   --causal_top_k $causal_top_k \
#   --model_comment $comment

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ./checkpoints/"
echo "=========================================="