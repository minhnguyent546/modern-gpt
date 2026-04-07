#!/usr/bin/env bash

export NUM_PROC_PER_NODE=2

export OMP_NUM_THREADS=4
export USE_FLASH_ATTN=1

# NOTE
# - warmup for 0.27B tokens

uv run torchrun --standalone --nproc_per_node=$NUM_PROC_PER_NODE -m modern_lm.train \
  --checkpoints_dir ./checkpoints \
  --seed 1061109567 \
  --train_dir ./data/Fineweb-Edu-10BT/train \
  --val_dir ./data/Fineweb-Edu-10BT/val \
  --vocab_size 49152 \
  --seq_length 2048 \
  --d_model 768 \
  --num_layers 12 \
  --num_heads 12 \
  --d_ff 2048 \
  --tie_weights \
  --rope_theta 10000.0 \
  --final_logit_softcapping 30.0 \
  --wandb_logging \
  --wandb_project modern-lm \
  --wandb_name v1-2026-03-31_07-00-01 \
  --wandb_logging_interval 15 \
  --matmul_precision high \
  --optim_type muon \
  --learning_rate 4e-4 \
  --muon_lr 0.01 \
  --adam_betas 0.9 0.999 \
  --adam_eps 1e-10 \
  --weight_decay 0.1 \
  --lr_schedule wsd \
  --min_lr 4e-5 \
  --train_steps 3814 \
  --val_steps 640 \
  --warmup_steps 500 \
  --stable_steps 2552 \
  --decay_steps 762 \
  --decay_type 1-sqrt \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --gradient_accum_step 32 \
  --mixed_precision bf16 \
  --val_interval 100 \
  --save_interval 1000 \
  --saved_checkpoint_limit 6 \
  --save_model_only \
  --log_interval 5 \
  --max_grad_norm 1.0
