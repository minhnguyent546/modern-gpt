import argparse


def add_run_pretrain_opts(parser: argparse.ArgumentParser) -> None:
    """Options for pre-training model."""
    _add_general_opts(parser)
    _add_dataset_opts(parser)
    _add_model_opts(parser)
    _add_wandb_opts(parser)
    _add_common_training_opts(parser)
    _add_ddp_training_opts(parser)


def _add_general_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("General")
    group.add_argument(
        "--checkpoints_dir",
        type=str,
        help="Directory to save model checkpoints",
        default="checkpoints",
    )
    group.add_argument(
        "--seed",
        type=int,
        help="Seed for random number generators",
        default=1061109567,
    )


def _add_dataset_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--train_dir",
        type=str,
        help="Directory contains training shards",
        default="fineweb_edu/train",
    )
    group.add_argument(
        "--val_dir",
        type=str,
        help="Directory contains validation shards",
        default="fineweb_edu/val",
    )


def _add_model_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Model")
    group.add_argument(
        "--vocab_size",
        type=int,
        help="Vocabulary size",
        default=50257,
    )
    group.add_argument(
        "--seq_length",
        type=int,
        help="Maximum sequence length",
        default=512,
    )
    group.add_argument(
        "--d_model",
        type=int,
        help="Size of the embedding vectors",
        default=768,
    )
    group.add_argument(
        "--num_layers",
        type=int,
        help="Number of hidden layers",
        default=12,
    )
    group.add_argument(
        "--num_heads",
        type=int,
        help="Number of attention heads",
        default=12,
    )
    group.add_argument(
        "--num_kv_heads",
        type=int,
        help="Number of key/value heads for Grouped-Query Attention",
        default=4,
    )
    group.add_argument(
        "--d_ff",
        type=int,
        help="Intermediate size of the feed-forward layers",
        default=2048,  # use 8/3 * d_model to achive the same number of parameters compare to FFN when switching to SwiGLU
    )
    group.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate",
        default=0.0,
    )
    group.add_argument(
        "--tie_weights",
        action="store_true",
        help="Whether to tie weights between input and output embeddings",
    )
    group.add_argument(
        "--rope_theta_full",
        type=float,
        help="Theta value for RoPE in full attention layers",
        default=100_000.0,
    )
    group.add_argument(
        "--rope_theta_sliding",
        type=float,
        help="Theta value for RoPE in sliding window attention layers",
        default=10_000.0,
    )
    group.add_argument(
        "--sliding_window_size",
        type=int,
        help="Sliding window size for attention",
        default=512,
    )
    group.add_argument(
        "--layer_types",
        type=str,
        nargs="+",
        help="List of layer types (e.g., 'sliding,sliding,sliding,sliding,full'). It will be repeated to match num_layers.",
        default=["sliding", "sliding", "sliding", "sliding", "full"],
    )
    group.add_argument(
        "--partial_rotary_factor",
        type=float,
        help="Fraction of head dimensions to apply RoPE to (1.0 = all)",
        default=1.0,
    )
    group.add_argument(
        "--attn_logit_softcapping",
        type=float,
        help="Softcapping value for attention logits",
        default=0.0,  # use QK-Norm instead
    )
    group.add_argument(
        "--final_logit_softcapping",
        type=float,
        help="Softcapping value for final logits",
        default=30.0,
    )
    group.add_argument(
        "--rms_norm_eps",
        type=float,
        help="Epsilon value for RMS normalization. Use larger value (e.g., 1e-5) for training stability when using mixed precision training and smaller value (e.g., 1e-7) for better performance when using full precision training",
        default=1e-5,
    )


def _add_wandb_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Wandb")
    group.add_argument(
        "--wandb_logging",
        action="store_true",
        help="Enable logging to wandb",
    )
    group.add_argument(
        "--wandb_project",
        type=str,
        help="Project name",
        default="modern-lm",
    )
    group.add_argument(
        "--wandb_name",
        type=str,
        help="Experiment name",
        default="v0-baseline",
    )
    group.add_argument(
        "--wandb_logging_interval",
        type=int,
        help="Time between syncing metrics to wandb",
        default=100,
    )
    group.add_argument(
        "--wandb_resume_id",
        type=str,
        help="Id to resume a run from",
    )
    group.add_argument(
        "--wandb_notes",
        type=str,
        help="Wandb notes",
    )
    group.add_argument(
        "--wandb_tags",
        type=str,
        help="Wandb tags",
    )


def _add_common_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Training")
    group.add_argument(
        "--run_evals_only",
        help="Run evaluation only",
        action="store_true",
    )
    group.add_argument(
        "--compile",
        help="Whether to compile the model with torch.compile",
        action="store_true",
    )
    group.add_argument(
        "--matmul_precision",
        type=str,
        help="Sets the internal precision of float32 matrix multiplications",
        choices=["highest", "high", "medium"],
        default="highest",
    )
    group.add_argument(
        "--gpu_peak_tflops",
        type=float,
        help="Peak bf16 tensor-core TFLOPS of the GPU (e.g. 989.5 for H100 SXM, 165.2 for RTX 4090). "
        "Used to compute MFU. Auto-detected for common GPUs if not provided.",
    )

    # optimizer options
    group.add_argument(
        "--optim_type",
        type=str,
        help="Which optimizer to use",
        choices=["adam", "adamw", "muon"],
        default="adamw",
    )
    group.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=6.0e-4,
    )
    group.add_argument(
        "--muon_lr",
        type=float,
        help="Learning rate for Muon optimizer",
        default=6.0e-3,
    )
    group.add_argument(
        "--adam_betas",
        nargs=2,
        type=float,
        help="Optimizer beta values",
        default=[0.9, 0.999],
    )
    group.add_argument(
        "--adam_eps",
        type=float,
        help="Optimizer epsilon value",
        default=1e-8,
    )
    group.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay value",
        default=0.0,
    )

    # scheduler options
    group.add_argument(
        "--lr_schedule",
        type=str,
        help="Learning rate scheduler (you might want to choose larger learning rate when using noam decay, e.g. 0.5)",
        choices=["cosine", "noam", "wsd"],
        default="wsd",
    )
    group.add_argument(
        "--warmup_steps",
        type=int,
        help="Warmup steps for learning rate",
        default=1_000,
    )
    group.add_argument(
        "--min_lr",
        type=float,
        help="Minimum learning rate (i.e. decay until this value) (for noam decay only)",
        default=6.0e-5,
    )
    group.add_argument(
        "--stable_steps",
        type=int,
        help="Number of steps to maintain constant learning rate (for wsd decay only). If not provided, the value will be inferred from --train_steps, --warup_steps and --decay_steps to make sure the learning rate will decay until min_lr at the end of training.",
    )
    group.add_argument(
        "--decay_steps",
        type=int,
        help="Number of steps to decay learning rate (for cosine decay only)",
        default=20_000,
    )
    group.add_argument(
        "--decay_type",
        type=str,
        help="Type of decay (for wsd decay only)",
        choices=["linear", "cosine", "1-sqrt"],
        default="1-sqrt",
    )

    # others
    group.add_argument(
        "--train_batch_size",
        type=int,
        help="Training batch size",
        default=8,
    )
    group.add_argument(
        "--eval_batch_size",
        type=int,
        help="Evaluation batch size",
        default=16,
    )
    group.add_argument(
        "--gradient_accum_step",
        type=int,
        help="Gradient accumulation step",
        default=32,
    )
    group.add_argument(
        "--mixed_precision",
        type=str,
        help="Data type for mixed precision training",
        choices=["fp16", "bf16"],
    )
    group.add_argument(
        "--train_steps",
        type=int,
        help="Number of training steps (i.e. number of optimizer steps)",
        default=20_000,
    )
    group.add_argument(
        "--val_steps",
        type=int,
        help="Number of validation steps",
        default=100,
    )
    group.add_argument(
        "--val_interval",
        type=int,
        help="Steps between validation",
        default=1_000,
    )
    group.add_argument(
        "--log_interval",
        type=int,
        help="Steps between logging to file",
        default=10,
    )
    group.add_argument(
        "--save_interval",
        type=int,
        help="Steps between saving checkpoints (you SHOULD use the SAME value as --val-interval for accurate training loss when resuming from previous checkpoint)",
        default=1_000,
    )
    group.add_argument(
        "--saved_checkpoint_limit",
        type=int,
        help="Maximum number of saved checkpoints, when reached, the oldest checkpoints will be removed",
        default=10,
    )
    group.add_argument(
        "--save_model_only",
        action="store_true",
        help="Only save the model weights, do not save optimizer and scheduler states.",
    )
    group.add_argument(
        "--max_grad_norm",
        type=float,
        help="Maximum gradient norm for gradient clipping (0.0 means no clipping)",
        default=0.0,
    )
    group.add_argument(
        "--from_checkpoint",
        type=str,
        help="Start the training from this saved checkpoint",
    )


def _add_ddp_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("DDP training")
    group.add_argument(
        "--ddp_backend",
        type=str,
        help="DDP backend used for distributed training",
        default="nccl",
    )
