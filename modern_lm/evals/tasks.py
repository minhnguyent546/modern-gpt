"""
Dataset-specific loaders and evaluation runners for multiple-choice benchmarks:
HellaSwag, OpenBookQA, ARC-Easy, ARC-Challenge.
"""

import re
import time
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import logging as hf_logging
from torch import nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer

from modern_lm.evals.mc_engine import NormalizationType, PackedSequence, pack_tasks, score_sequence

hf_logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
EOT_ID = tokenizer.eos_token_id


def _preprocess_hellaswag_text(text: str) -> str:
    """Preprocessing from AI-Harness for WikiHow artifacts in HellaSwag."""
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def _preprocess_hellaswag(raw_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for raw_task in raw_tasks:
        ctx = f"{raw_task['ctx_a']} {raw_task['ctx_b'].capitalize()} "
        query_tokens = tokenizer(
            _preprocess_hellaswag_text(raw_task["activity_label"] + ": " + ctx),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
        preprocessed_endings = [
            _preprocess_hellaswag_text(ending) for ending in raw_task["endings"]
        ]
        choices_tokens = tokenizer(
            [" " + ending for ending in preprocessed_endings],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
        items.append({
            "query_tokens": query_tokens,
            "choices_tokens": choices_tokens,
            "choice_char_lengths": [len(ending) for ending in preprocessed_endings],
            "label": int(raw_task["label"]),
        })
    return items


def _preprocess_openbookqa(raw_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for raw_task in raw_tasks:
        query_tokens = tokenizer(
            raw_task["question_stem"],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
        choices = raw_task["choices"]
        choice_texts = choices["text"]
        choices_tokens = tokenizer(
            [" " + text for text in choice_texts],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
        label = choices["label"].index(raw_task["answerKey"])
        items.append({
            "query_tokens": query_tokens,
            "choices_tokens": choices_tokens,
            "choice_char_lengths": [len(text) for text in choice_texts],
            "label": label,
        })
    return items


def _preprocess_arc(raw_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for raw_task in raw_tasks:
        query_tokens = tokenizer(
            f"Question: {raw_task['question']}\nAnswer:",
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
        choices = raw_task["choices"]
        choice_texts = choices["text"]
        choices_tokens = tokenizer(
            [" " + text for text in choice_texts],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
        label = choices["label"].index(raw_task["answerKey"].strip())
        items.append({
            "query_tokens": query_tokens,
            "choices_tokens": choices_tokens,
            "choice_char_lengths": [len(text) for text in choice_texts],
            "label": label,
        })
    return items


@lru_cache(1)
def _get_hellaswag_sequences(seq_len: int, rank: int, world_size: int) -> list[PackedSequence]:
    dataset = load_dataset(
        path="Rowan/hellaswag",
        name="default",
        split="validation",
        revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
    )
    dataset = dataset.shuffle(seed=42)
    raw_tasks = [dataset[i + rank] for i in range(0, len(dataset) - rank, world_size)]
    items = _preprocess_hellaswag(raw_tasks)
    return pack_tasks(
        items,
        seq_len=seq_len,
        eot_id=EOT_ID,
        doc_len_min_average=70,  # use a smaller number than the median doc length of 88 (doc=context+one ending)
    )


@lru_cache(1)
def _get_obqa_sequences(seq_len: int, rank: int, world_size: int) -> list[PackedSequence]:
    dataset = load_dataset(
        path="allenai/openbookqa",
        name="main",
        split="validation",
        revision="388097ea7776314e93a529163e0fea805b8a6454",
    )
    dataset = dataset.shuffle(seed=42)
    raw_tasks = [dataset[i + rank] for i in range(0, len(dataset) - rank, world_size)]
    items = _preprocess_openbookqa(raw_tasks)
    return pack_tasks(
        items,
        seq_len=seq_len,
        eot_id=EOT_ID,
        doc_len_min_average=10,  # use a smaller number than the median doc length of 14 (doc=context+one ending)
    )


@lru_cache(1)
def _get_arc_easy_sequences(seq_len: int, rank: int, world_size: int) -> list[PackedSequence]:
    dataset = load_dataset(
        path="allenai/ai2_arc",
        name="ARC-Easy",
        split="test",
        revision="210d026faf9955653af8916fad021475a3f00453",
    )
    dataset = dataset.shuffle(seed=42)
    raw_tasks = [dataset[i + rank] for i in range(0, len(dataset) - rank, world_size)]
    items = _preprocess_arc(raw_tasks)
    return pack_tasks(
        items,
        seq_len=seq_len,
        eot_id=EOT_ID,
        doc_len_min_average=10,  # use a smaller number than the median doc length of 27 (doc=context+one ending)
    )


@lru_cache(1)
def _get_arc_challenge_sequences(seq_len: int, rank: int, world_size: int) -> list[PackedSequence]:
    dataset = load_dataset(
        path="allenai/ai2_arc",
        name="ARC-Challenge",
        split="test",
        revision="210d026faf9955653af8916fad021475a3f00453",
    )
    dataset = dataset.shuffle(seed=42)
    raw_tasks = [dataset[i + rank] for i in range(0, len(dataset) - rank, world_size)]
    items = _preprocess_arc(raw_tasks)
    return pack_tasks(
        items,
        seq_len=seq_len,
        eot_id=EOT_ID,
        doc_len_min_average=10,  # use a smaller number than the median doc length of 33 (doc=context+one ending)
    )


def _score_benchmark(
    model: nn.Module,
    sequences: list[PackedSequence],
    normalization: NormalizationType,
    desc: str,
    show_progress_bar: bool = False,
    autocast_context=None,
) -> tuple[int, int]:
    n_correct, n_count = 0, 0
    progress_bar = tqdm(sequences, desc=f"Eval {desc}", disable=not show_progress_bar)
    for sequence in progress_bar:
        _correct, _count = score_sequence(
            model,
            sequence,
            autocast_context=autocast_context,
            normalization=normalization,
        )
        n_correct += _correct
        n_count += _count

    n_correct_tensor = torch.tensor([n_correct], device="cuda")
    n_count_tensor = torch.tensor([n_count], device="cuda")

    if dist.is_initialized():
        dist.all_reduce(n_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_count_tensor, op=dist.ReduceOp.SUM)
    return n_correct_tensor.item(), n_count_tensor.item()  # pyright: ignore[reportReturnType]


def _run_eval(
    model: nn.Module,
    seq_len: int,
    rank: int,
    world_size: int,
    get_sequences_fn,
    normalization: NormalizationType,
    desc: str,
    show_progress_bar: bool = False,
    autocast_context=None,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    is_training = model.training
    model.eval()
    with torch.inference_mode():
        sequences = get_sequences_fn(seq_len=seq_len, rank=rank, world_size=world_size)
        n_correct, n_count = _score_benchmark(
            model,
            sequences,
            normalization=normalization,
            desc=desc,
            show_progress_bar=show_progress_bar,
            autocast_context=autocast_context,
        )
        accuracy = n_correct / n_count

    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    model.train(is_training)

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_count": n_count,
        "seconds": seconds,
    }


def run_eval_hellaswag(
    model: nn.Module,
    seq_len: int,
    rank: int,
    world_size: int,
    show_progress_bar: bool = False,
    autocast_context=None,
) -> dict[str, Any]:
    """Evaluate on HellaSwag validation set (10 042 tasks, acc_norm)."""
    return _run_eval(
        model,
        seq_len,
        rank,
        world_size,
        get_sequences_fn=_get_hellaswag_sequences,
        normalization="acc_norm",
        desc="HellaSwag",
        show_progress_bar=show_progress_bar,
        autocast_context=autocast_context,
    )


def run_eval_obqa(
    model: nn.Module,
    seq_len: int,
    rank: int,
    world_size: int,
    show_progress_bar: bool = False,
    autocast_context=None,
) -> dict[str, Any]:
    """Evaluate on OpenBookQA test set (500 tasks, acc_norm)."""
    return _run_eval(
        model,
        seq_len,
        rank,
        world_size,
        get_sequences_fn=_get_obqa_sequences,
        normalization="acc_norm",
        desc="OpenBookQA",
        show_progress_bar=show_progress_bar,
        autocast_context=autocast_context,
    )


def run_eval_arc_easy(
    model: nn.Module,
    seq_len: int,
    rank: int,
    world_size: int,
    show_progress_bar: bool = False,
    autocast_context=None,
) -> dict[str, Any]:
    """Evaluate on ARC-Easy test set (2 376 tasks, acc_norm)."""
    return _run_eval(
        model,
        seq_len,
        rank,
        world_size,
        get_sequences_fn=_get_arc_easy_sequences,
        normalization="acc_norm",
        desc="ARC-Easy",
        show_progress_bar=show_progress_bar,
        autocast_context=autocast_context,
    )


def run_eval_arc_challenge(
    model: nn.Module,
    seq_len: int,
    rank: int,
    world_size: int,
    show_progress_bar: bool = False,
    autocast_context=None,
) -> dict[str, Any]:
    """Evaluate on ARC-Challenge test set (1 172 tasks, acc_norm)."""
    return _run_eval(
        model,
        seq_len,
        rank,
        world_size,
        get_sequences_fn=_get_arc_challenge_sequences,
        normalization="acc_norm",
        desc="ARC-Challenge",
        show_progress_bar=show_progress_bar,
        autocast_context=autocast_context,
    )
