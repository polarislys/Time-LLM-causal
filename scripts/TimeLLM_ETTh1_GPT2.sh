#!/bin/bash
# 切换到 Time-LLM 目录
cd "$(dirname "$0")/.." || exit 1
# === 缓存路径配置 ===
export HF_HOME="/home/nl/disk_8T/lys/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HF_HOME/torch"
export TMPDIR="/tmp"
export HF_ENDPOINT="https://hf-mirror.com"  # 使用镜像站
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 🔥 修复1：强制限制 CPU 线程数（解决 11 个进程假象和 CPU 抢占）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 创建缓存目录
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TMPDIR"

# === 模型参数 ===
model_name=TimeLLM
train_epochs=15
learning_rate=0.01
gpt2_layers=6

# === 训练配置 ===
master_port=0
num_process=1  # 使用单个GPU
batch_size=8
d_model=32
d_ff=128

comment='TimeLLM-ETTh1-GPT2'

# === 因果模块参数 ===
use_causal="--use_causal"
use_amp="--use_amp" 
causal_cache_dir="./causal_results"
causal_tau_max=5
causal_pc_alpha=0.01
causal_top_k=5

use_causal_loss="--use_causal_loss"
causal_loss_weight=0.1


accelerate launch --mixed_precision fp16 --num_processes 1 --num_machines 1 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
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
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers $gpt2_layers \
  --llm_cache_dir $HF_HOME \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --num_workers 0 \
  $use_causal \
  $use_causal_loss \
  $use_amp \
  --causal_cache_dir $causal_cache_dir \
  --causal_tau_max $causal_tau_max \
  --causal_pc_alpha $causal_pc_alpha \
  --causal_top_k $causal_top_k \
  --causal_loss_weight $causal_loss_weight
  

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ./checkpoints/"
echo "=========================================="
