import argparse
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.version
import wandb
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.autonotebook import tqdm

import modern_lm.opts as opts
import modern_lm.utils as utils
from modern_lm.evals import (
    run_eval_arc_challenge,
    run_eval_arc_easy,
    run_eval_hellaswag,
    run_eval_obqa,
)
from modern_lm.lm_dataset import LMDataset
from modern_lm.meters import AverageMeter
from modern_lm.model import ModernLM, ModernLMConfig


def train_model(args: argparse.Namespace) -> None:
    # set seed
    utils.set_seed(args.seed)

    # checkpoint dir and log file
    checkpoints_dir = None
    log_file = None
    if not args.run_evals_only:
        checkpoints_dir_basename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.wandb_logging and args.wandb_name is not None:
            checkpoints_dir_basename += f"-{args.wandb_name}"

        checkpoints_dir = os.path.join(
            args.checkpoints_dir,
            checkpoints_dir_basename,
        )
        os.makedirs(checkpoints_dir, exist_ok=True)
        if args.wandb_logging and args.wandb_name is not None:
            log_file = os.path.join(checkpoints_dir, f"{args.wandb_name}.log")
        else:
            log_file = os.path.join(checkpoints_dir, "training.log")

    def master_print(message: str, console: bool = True) -> None:
        if args.is_master and not args.run_evals_only and log_file is not None:
            if console:
                print(message)
            with open(log_file, "a") as f:
                print(message, file=f)

    git_info = utils.get_git_info()
    if git_info.get("repository") is not None:
        master_print(f"Git repository: {git_info['repository']}")
    if git_info.get("commit_hash") is not None:
        master_print(f"Git commit hash: {git_info['commit_hash']}")
    if git_info.get("branch") is not None:
        master_print(f"Git branch: {git_info['branch']}")
    master_print(f"Python version: {sys.version}")
    master_print(
        f"Pytorch version {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
    )
    master_print(
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout
    )
    master_print(f"Args: {vars(args)}")

    # training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    master_print(f"Using device: {device}")

    torch.set_float32_matmul_precision(args.matmul_precision)
    master_print(f"Set float32 matmul precision to {args.matmul_precision}")

    if args.train_batch_size % args.world_size != 0:
        raise ValueError("train_batch_size must be divisible by world_size")
    if args.eval_batch_size % args.world_size != 0:
        raise ValueError("eval_batch_size must be divisible by world_size")
    train_batch_size = args.train_batch_size // args.world_size
    eval_batch_size = args.eval_batch_size // args.world_size
    effective_batch_size = train_batch_size * args.world_size * args.gradient_accum_step
    tokens_per_fwdbwd = (
        train_batch_size * args.world_size * args.seq_length * args.gradient_accum_step
    )
    master_print(
        f"Effective batch size: {effective_batch_size} "
        f"(micro_batch_size={train_batch_size}, "
        f"gradient_accum_step={args.gradient_accum_step}, "
        f"num_devices={args.world_size})"
    )
    master_print(f"Tokens per forward/backward pass: {tokens_per_fwdbwd}")

    # dataset
    train_dataset = LMDataset(
        args.train_dir,
        train_batch_size,
        args.seq_length,
        num_replicas=args.world_size,
        rank=args.rank,
    )
    val_dataset = LMDataset(
        args.val_dir,
        eval_batch_size,
        args.seq_length,
        num_replicas=args.world_size,
        rank=args.rank,
    )
    if train_dataset.total_tokens is not None:
        master_print(f"Total tokens in training dataset: {train_dataset.total_tokens}")
    if val_dataset.total_tokens is not None:
        master_print(f"Total tokens in validation dataset: {val_dataset.total_tokens}")

    # logging with wandb
    wandb_run = None
    if args.is_master and args.wandb_logging and not args.run_evals_only:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            id=args.wandb_resume_id,
            resume="must" if args.wandb_resume_id is not None else None,
        )
        master_print(
            f"Wandb logging enabled. project: {args.wandb_project}, name: {args.wandb_name}, id: {wandb_run.id}"
        )

    # mixed precision training
    mp_dtype = torch.float32
    if device.type == "cuda":
        if args.mixed_precision == "fp16":
            mp_dtype = torch.float16
            master_print("Mixed precision training is enabled with fp16")
        elif args.mixed_precision == "bf16":
            if torch.cuda.is_bf16_supported():
                mp_dtype = torch.bfloat16
                master_print("Mixed precision training is enabled with bf16")
            else:
                mp_dtype = torch.float16
                master_print("bf16 is not supported on your hardware, fallback to fp16")
    autocast_context = torch.amp.autocast_mode.autocast(
        device_type=device.type,
        dtype=mp_dtype,
        enabled=(mp_dtype in (torch.float16, torch.bfloat16)),
    )
    autocast_enabled = autocast_context._enabled  # pyright: ignore[reportPrivateUsage]
    if not autocast_enabled:
        autocast_context = nullcontext()

    scaler = torch.amp.grad_scaler.GradScaler(
        device=device.type, enabled=(mp_dtype == torch.float16)
    )

    # resume from previous checkpoint
    saved_states = None
    if args.from_checkpoint is None:
        modern_lm_config = ModernLMConfig(
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            tie_weights=args.tie_weights,
            attn_logit_softcapping=args.attn_logit_softcapping,
            final_logit_softcapping=args.final_logit_softcapping,
        )
        model = ModernLM(modern_lm_config)
    else:
        master_print(f"Loading states from checkpoint {args.from_checkpoint}")
        saved_states = torch.load(args.from_checkpoint, map_location=device)
        required_keys = ["model", "optimizer", "lr_scheduler", "config"]
        if scaler.is_enabled():
            required_keys.append("scaler")
        for key in required_keys:
            if key not in saved_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')
        modern_lm_config = ModernLMConfig(**saved_states["config"])
        model = ModernLM(modern_lm_config)

    model.to(device)
    if model.config.tie_weights:
        model.tie_weights()

    if os.getenv("USE_FLASH_ATTN") == "1":
        master_print("USE_FLASH_ATTN is set, trying to use flash attention if available")

    master_print(f"Model: {model}")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    eval_criterion = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    optimizer = utils.make_optimizer(
        model=model,
        device=device,
        optim_type=args.optim_type,
        lr=learning_rate,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
    )
    if args.lr_schedule == "noam":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.noam_decay(
                step,
                args.d_model,
                args.warmup_steps,
            ),
        )
    elif args.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.cosine_decay(
                step,
                learning_rate,
                args.min_lr,
                args.warmup_steps,
                args.decay_steps,
                factor=1 / learning_rate,
            ),
        )
    elif args.lr_schedule == "wsd":
        lr_scheduler = utils.get_wsd_schedule(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_stable_steps=args.stable_steps,
            num_decay_steps=args.decay_steps,
            min_lr_ratio=args.min_lr / learning_rate,
            decay_type=args.decay_type,
        )
    else:
        raise ValueError(f"Unsupported scheduler decay method: {args.lr_schedule}")

    initial_step = 0
    if saved_states is not None:
        unwanted_prefixes = ["module.", "_orig_mod."]  # created by DDP() and torch.compile()
        for prefix in unwanted_prefixes:
            consume_prefix_in_state_dict_if_present(saved_states["model"], prefix=prefix)

        model.load_state_dict(saved_states["model"])
        optimizer.load_state_dict(saved_states["optimizer"])
        lr_scheduler.load_state_dict(saved_states["lr_scheduler"])
        if scaler.is_enabled():
            scaler.load_state_dict(saved_states["scaler"])
        if "global_step" in saved_states:
            initial_step = saved_states["global_step"]

    raw_model = model
    # compile the model
    if args.compile:
        master_print("Compiling the model")
        model = torch.compile(model, dynamic=False, fullgraph=True)

    # wrap the model with DDP
    if args.ddp_enabled:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            gradient_as_bucket_view=True,
        )

    if args.run_evals_only:
        val_results = eval_model(
            model=model,
            device=device,
            criterion=eval_criterion,
            eval_dataset=val_dataset,
            eval_steps=args.val_steps,
            args=args,
            autocast_context=autocast_context,
            show_progress_bar=True,
        )

        eval_kwargs = {
            "model": model,
            "seq_len": raw_model.config.seq_length,
            "rank": args.rank,
            "world_size": args.world_size,
            "show_progress_bar": True,
            "autocast_context": autocast_context,
        }
        hellaswag_result = run_eval_hellaswag(**eval_kwargs)  # pyright: ignore[reportArgumentType]
        obqa_result = run_eval_obqa(**eval_kwargs)  # pyright: ignore[reportArgumentType]
        arc_easy_result = run_eval_arc_easy(**eval_kwargs)  # pyright: ignore[reportArgumentType]
        arc_challenge_result = run_eval_arc_challenge(**eval_kwargs)  # pyright: ignore[reportArgumentType]

        if args.is_master:
            print("** Evaluation results **")
            print(f"Loss: {val_results['loss']}")
            print(f"Number of evaluation tokens: {val_results['num_eval_tokens']}")
            print(f"Perplexity: {utils.get_perplexity(val_results['loss'])}")
            for name, result in [
                ("HellaSwag", hellaswag_result),
                ("OpenBookQA", obqa_result),
                ("ARC-Easy", arc_easy_result),
                ("ARC-Challenge", arc_challenge_result),
            ]:
                print(
                    f"{name}: {result['accuracy']:0.3%} "
                    f"({result['n_correct']} / {result['n_count']} tasks "
                    f"in {utils.to_hms(result['seconds'])})"
                )
        return

    assert checkpoints_dir is not None
    flops_per_token = raw_model.estimate_flops_per_token()
    gpu_peak_flops: float | None = None
    if getattr(args, "gpu_peak_tflops", None) is not None and args.gpu_peak_tflops > 0:
        gpu_peak_flops = args.gpu_peak_tflops * 1e12
    else:
        gpu_peak_flops = utils.get_gpu_peak_flops(torch.cuda.get_device_name(device))
    if args.is_master:
        num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
        master_print(f"Model has {num_parameters / 10**6:0.2f}M parameters")
        master_print(f"Estimated FLOPs per token (fwd+bwd): {flops_per_token:.2e}")
        if gpu_peak_flops is not None:
            master_print(f"GPU peak FLOPS (bf16): {gpu_peak_flops:.2e}")
        else:
            master_print(
                "GPU peak FLOPS unknown — MFU will not be reported. "
                "Use --gpu_peak_tflops to specify."
            )

    if args.ddp_enabled:
        train_iter = tqdm(
            range(initial_step, args.train_steps),
            desc=f"GPU{args.rank}-Training",
            disable=(not args.is_local_master),
            ncols=120,
        )
    else:
        train_iter = tqdm(
            range(initial_step, args.train_steps),
            desc="Training",
            ncols=120,
        )

    # training loop
    global_step = initial_step
    wandb_accum_logs: list[dict[str, Any]] = []
    running_loss = AverageMeter("running_loss", device=device)
    token_seen: int = 0

    # set model in training mode
    model.train()
    optimizer.zero_grad()
    train_loader_iter = iter(train_dataset)

    # Pre-fetch the first batch
    try:
        input_ids, labels = next(train_loader_iter)
    except StopIteration:
        train_loader_iter = iter(train_dataset)
        input_ids, labels = next(train_loader_iter)

    if input_ids.dim() == 3:
        assert input_ids.shape[0] == 1
        input_ids = input_ids[0]
    if labels.dim() == 3:
        assert labels.shape[0] == 1
        labels = labels[0]

    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    training_start_time = time.perf_counter()
    while global_step < args.train_steps:
        last_step = global_step + 1 >= args.train_steps
        num_items_in_batch = torch.tensor(0, device=device)
        batch_loss = 0.0
        step_start_time = time.perf_counter()
        for batch_idx in range(args.gradient_accum_step):
            # TODO: assume padding token id is -100, replace with actual padding token id if different
            num_items_in_batch += (labels != -100).sum()

            if args.ddp_enabled:
                # we only sync gradients at the last step of gradient accumulation
                # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
                model.require_backward_grad_sync = batch_idx + 1 == args.gradient_accum_step

            with autocast_context:
                logits = model(input_ids)
                loss = criterion(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))

            # Fetch the next batch from CPU and start async transfer to GPU while forward/backward is executing
            try:
                next_input_ids, next_labels = next(train_loader_iter)
            except StopIteration:
                master_print(
                    f"DataLoader is exhausted at step {global_step}, restarting the DataLoader for the next epoch."
                )
                train_loader_iter = iter(train_dataset)
                next_input_ids, next_labels = next(train_loader_iter)

            if next_input_ids.dim() == 3:
                assert next_input_ids.shape[0] == 1
                next_input_ids = next_input_ids[0]
            if next_labels.dim() == 3:
                assert next_labels.shape[0] == 1
                next_labels = next_labels[0]

            next_input_ids = next_input_ids.to(device, non_blocking=True)
            next_labels = next_labels.to(device, non_blocking=True)

            scaler.scale(loss).backward()
            batch_loss += loss.detach()

            # Swap buffers for the next micro-step
            input_ids = next_input_ids
            labels = next_labels

        batch_loss = batch_loss / num_items_in_batch
        token_seen += tokens_per_fwdbwd

        scaler.unscale_(optimizer)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(num_items_in_batch)

        grad_norm_value = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=(args.max_grad_norm if args.max_grad_norm > 0 else float("inf")),
            norm_type=2,
        )
        if bool(torch.isinf(grad_norm_value)) or bool(torch.isnan(grad_norm_value)):
            grad_norm_value = -1

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start_time
        batch_throughput = tokens_per_fwdbwd / step_time

        mfu = -1.0
        if gpu_peak_flops is not None and step_time > 0:
            per_device_tokens = tokens_per_fwdbwd / args.world_size
            mfu = flops_per_token * per_device_tokens / step_time / gpu_peak_flops

        # TODO: handle the case when wandb is disabled
        wandb_accum_logs.append({
            f"learning_rate/group_{group_id}": group_lr
            for group_id, group_lr in enumerate(lr_scheduler.get_last_lr())
        })
        wandb_accum_logs[-1].update({
            "loss/batch_loss": batch_loss,
            "grad_norm": grad_norm_value,
            "throughput": batch_throughput,
            "mfu": mfu,
            "token_seen": token_seen,
            "step": global_step,
        })

        lr_scheduler.step()
        running_loss.update(batch_loss, num_items_in_batch)  # pyright: ignore[reportArgumentType]

        # run validation
        if (global_step + 1) % args.val_interval == 0 or last_step:
            if args.ddp_enabled:
                running_loss.reduce(dst=args.master_rank)
            val_results = eval_model(
                model=model,
                device=device,
                criterion=eval_criterion,
                eval_dataset=val_dataset,
                eval_steps=args.val_steps,
                args=args,
                autocast_context=autocast_context,
            )

            eval_kwargs = {
                "model": model,
                "seq_len": raw_model.config.seq_length,
                "rank": args.rank,
                "world_size": args.world_size,
                "autocast_context": autocast_context,
            }
            hellaswag_result = run_eval_hellaswag(**eval_kwargs)  # pyright: ignore[reportArgumentType]
            obqa_result = run_eval_obqa(**eval_kwargs)  # pyright: ignore[reportArgumentType]
            arc_easy_result = run_eval_arc_easy(**eval_kwargs)  # pyright: ignore[reportArgumentType]
            arc_challenge_result = run_eval_arc_challenge(**eval_kwargs)  # pyright: ignore[reportArgumentType]
            wandb_accum_logs[-1].update({
                "loss/train": running_loss.average,
                "loss/val": val_results["loss"],
                "val/hellaswag": hellaswag_result["accuracy"],
                "val/obqa": obqa_result["accuracy"],
                "val/arc_easy": arc_easy_result["accuracy"],
                "val/arc_challenge": arc_challenge_result["accuracy"],
            })
            master_print(
                f"[step {global_step + 1} / {args.train_steps}] running_loss: {running_loss.average:0.4f} | "
                f"val_loss: {val_results['loss']:0.4f} | "
                f"num_eval_tokens: {val_results['num_eval_tokens']} | "
                f"hellaswag: {hellaswag_result['accuracy']:0.4f} | "
                f"obqa: {obqa_result['accuracy']:0.4f} | "
                f"arc_easy: {arc_easy_result['accuracy']:0.4f} | "
                f"arc_challenge: {arc_challenge_result['accuracy']:0.4f}"
            )
            running_loss.reset()

        # log to wandb
        if len(wandb_accum_logs) >= args.wandb_logging_interval or (
            len(wandb_accum_logs) > 0 and last_step
        ):
            if args.ddp_enabled:
                batch_loss_values = torch.tensor(
                    [loss["loss/batch_loss"] for loss in wandb_accum_logs],
                    dtype=torch.float32,
                    device=device,
                )
                dist.all_reduce(batch_loss_values, op=dist.ReduceOp.AVG)
                reduced_batch_loss_values = batch_loss_values.tolist()
                for idx in range(len(wandb_accum_logs)):
                    wandb_accum_logs[idx]["loss/batch_loss"] = reduced_batch_loss_values[idx]
            if wandb_run is not None:
                for log_idx in range(len(wandb_accum_logs)):
                    wandb_run.log(wandb_accum_logs[log_idx])
            wandb_accum_logs = []

            if args.ddp_enabled:
                dist.barrier()

        # save checkpoint
        if (global_step + 1) % args.save_interval == 0 or last_step:
            if args.is_master:
                checkpoint_dict = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": vars(modern_lm_config),
                    "global_step": global_step + 1,
                }
                if scaler.is_enabled():
                    checkpoint_dict["scaler"] = scaler.state_dict()
                utils.ensure_num_saved_checkpoints(
                    checkpoints_dir=args.checkpoints_dir,
                    model_basename="modern-lm",
                    limit=args.saved_checkpoint_limit,
                )
                model_save_path = os.path.join(checkpoints_dir, f"modern-lm-{global_step + 1}.pt")
                torch.save(checkpoint_dict, model_save_path)

            if args.ddp_enabled:
                dist.barrier()

        if (global_step + 1) % args.log_interval == 0 or last_step:
            mfu_str = f" | mfu: {mfu:0.1%}" if mfu >= 0 else ""
            master_print(
                f"\n[step {global_step + 1} / {args.train_steps}] loss: {batch_loss:0.4f} | throughput: {batch_throughput:0.1f} tokens/s | grad_norm: {grad_norm_value:0.4f}{mfu_str} | token_seen: {token_seen:0.2e}",
                console=False,
            )

        postfix = {
            "loss": f"{batch_loss:0.3f}",
            "grad_norm": f"{grad_norm_value:0.3f}",
            "token_seen": f"{token_seen:0.2e}",
        }
        if mfu >= 0:
            postfix["mfu"] = f"{mfu:0.1%}"
        train_iter.set_postfix(postfix)
        global_step += 1
        train_iter.update()

    training_time = time.perf_counter() - training_start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    master_print(
        f"*** Training stats: ***\n"
        f"  - Training time {utils.to_hms(training_time)}\n"
        f"  - Num tokens seen: {token_seen:0.2e}\n"
        f"  - Peak VRAM usage: {peak_vram_mb:.2f} MB\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opts.add_run_pretrain_opts(parser)
    args = parser.parse_args()

    setup_ddp(args)

    train_model(args)

    cleanup_ddp(args)


@torch.no_grad()
def eval_model(
    model: ModernLM | DDP,
    device: torch.device,
    criterion,
    eval_dataset,
    eval_steps: int,
    args: argparse.Namespace,
    autocast_context=None,
    show_progress_bar: bool = False,
) -> dict[str, float]:
    evaluation_loss = AverageMeter("evaluation_loss", device=device)
    if autocast_context is None:
        autocast_context = nullcontext()

    if args.ddp_enabled:
        progress_bar = tqdm(
            total=eval_steps,
            desc=f"GPU{args.rank}-Eval",
            disable=(not args.is_local_master) or (not show_progress_bar),
            position=1,
            leave=False,
            ncols=120,
        )
    else:
        progress_bar = tqdm(
            total=eval_steps,
            desc="Evaluating",
            disabled=(not show_progress_bar),
            position=1,
            leave=False,
            ncols=120,
        )

    # set model in evaluation mode
    is_training = model.training
    model.eval()
    num_eval_tokens = 0
    eval_loader_iter = iter(eval_dataset)
    for _ in range(eval_steps):
        try:
            input_ids, labels = next(eval_loader_iter)
        except StopIteration:
            break

        if input_ids.dim() == 3:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        if labels.dim() == 3:
            assert labels.shape[0] == 1
            labels = labels[0]

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context:
            logits = model(input_ids)
            loss = criterion(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))

        num_items_in_batch = (labels != -100).sum()
        num_eval_tokens += num_items_in_batch
        evaluation_loss.update(loss.detach(), num_items_in_batch)
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss:0.3f}"})

    # set model back to the original mode
    model.train(is_training)

    if args.ddp_enabled:
        evaluation_loss.reduce(dst=args.master_rank)
        num_eval_tokens = num_eval_tokens * args.world_size

    progress_bar.close()

    return {
        "loss": evaluation_loss.average,
        "num_eval_tokens": num_eval_tokens,
    }


def setup_ddp(args: argparse.Namespace) -> None:
    args.rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp_enabled = os.environ.get("RANK", -1) != -1
    args.master_rank = 0
    args.local_master_rank = 0
    args.is_master = args.rank == args.master_rank
    args.is_local_master = args.local_rank == args.local_master_rank
    if args.ddp_enabled:
        # set appropriate CUDA device
        torch.cuda.set_device(args.local_rank)
        # init process group
        dist.init_process_group(backend=getattr(args, "ddp_backend", "nccl"))  # nccl, gloo, etc


def cleanup_ddp(args: argparse.Namespace) -> None:
    if args.ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
