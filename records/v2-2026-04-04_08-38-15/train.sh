#!/usr/bin/env bash

export NUM_PROC_PER_NODE=2

export OMP_NUM_THREADS=4
export USE_FLASH_ATTN=1

# NOTE
# - warmup for 0.27B tokens

uv run torchrun --standalone --nproc_per_node=$NUM_PROC_PER_NODE -m modern_lm.train \
  --compile \
  --checkpoints_dir ./checkpoints \
  --seed 1061109567 \
  --train_dir ./data/Fineweb-Edu-10BT/train \
  --val_dir ./data/Fineweb-Edu-10BT/val \
  --vocab_size 49152 \
  --seq_length 8192 \
  --d_model 576 \
  --num_layers 30 \
  --num_heads 9 \
  --num_kv_heads 3 \
  --d_ff 1536 \
  --tie_weights \
  --final_logit_softcapping 30.0 \
  --rms_norm_eps 1.0e-05 \
  --rope_theta_full 100000.0 \
  --rope_theta_sliding 10000.0 \
  --sliding_window_size 512 \
  --layer_types sliding sliding sliding sliding full \
  --wandb_logging \
  --wandb_project modern-lm \
  --wandb_name v2-2026-04-04_08-38-15 \
  --wandb_logging_interval 15 \
  --matmul_precision high \
  --optim_type muon \
  --learning_rate 3e-3 \
  --muon_lr 0.02 \
  --adam_betas 0.9 0.999 \
  --adam_eps 1e-10 \
  --weight_decay 0.1 \
  --lr_schedule wsd \
  --min_lr 3e-4 \
  --train_steps 3814 \
  --val_steps 640 \
  --warmup_steps 500 \
  --stable_steps 2552 \
  --decay_steps 762 \
  --decay_type 1-sqrt \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accum_step 32 \
  --mixed_precision bf16 \
  --val_interval 100 \
  --save_interval 1000 \
  --saved_checkpoint_limit 6 \
  --save_model_only \
  --log_interval 5 \
  --max_grad_norm 1.0
