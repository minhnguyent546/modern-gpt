"""
Shared multiple-choice evaluation engine for LM benchmarks.

Supports tasks with variable number of choices (e.g. ARC has 3-5)
and configurable loss normalization (character-level for acc_norm, sum for acc).
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as Fun
from torch import Tensor, nn

NormalizationType = Literal["acc_norm", "acc"]


@dataclass
class MultipleChoiceTask:
    spans: list[tuple[int, int]]
    label: int
    choice_char_lengths: list[int]


@dataclass
class PackedSequence:
    """Contains multiple MC tasks packed into a single token sequence for efficient evaluation.

    Packing layout (same as the original HellaSwag packing):
      <ctx1><choice1>EOT<ctx1><choice2>EOT...<ctx2><choice1>EOT...

    self.inputs omits the last token of each choice (not needed for input).
    self.targets omits the first token of each context (not needed for target).
    """

    inputs: Tensor
    targets: Tensor
    tasks: list[MultipleChoiceTask]
    doc_end_positions: Tensor


def finalize_sequence(
    inputs: list[int],
    targets: list[int],
    tasks: list[MultipleChoiceTask],
    seq_len: int,
    eot_id: int,
    doc_len_min_average: int = 70,
) -> PackedSequence:
    max_num_docs = ((seq_len // doc_len_min_average) // 128 + 1) * 128
    # can use any token id except eot_id which delimits tasks/choices
    pad_id = 0 if eot_id != 0 else 1

    padding_length = seq_len - len(inputs)
    inputs.extend([pad_id] * padding_length)
    targets.extend([pad_id] * padding_length)
    assert len(inputs) == len(targets) == seq_len

    inputs_tensor = torch.tensor(inputs, dtype=torch.int32)
    targets_tensor = torch.tensor(targets, dtype=torch.int64)

    cum_lengths = torch.nonzero(inputs_tensor == eot_id)[:, 0] + 1
    _cum_lengths = torch.full((max_num_docs,), seq_len)
    _cum_lengths[0] = 0
    _cum_lengths[1 : len(cum_lengths) + 1] = cum_lengths
    _cum_lengths = _cum_lengths.to(dtype=torch.int32)

    return PackedSequence(
        inputs=inputs_tensor,
        targets=targets_tensor,
        tasks=tasks,
        doc_end_positions=_cum_lengths,
    )


def pack_tasks(
    items: list[dict[str, Any]],
    seq_len: int,
    eot_id: int,
    doc_len_min_average: int = 70,
) -> list[PackedSequence]:
    """Pack preprocessed MC items into sequences.

    The tasks are packed into a sequence like this:
    <ctx1><end11>EOT_ID<ctx1><end12>EOT_ID<ctx1><end13>EOT_ID<ctx1><end14>EOT_ID<ctx2><end21>...

    Each item must have:
      - query_tokens: list[int]
      - choices_tokens: list[list[int]]
      - choice_char_lengths: list[int]  (character length of each choice, excluding leading space)
      - label: int
    """
    sequences: list[PackedSequence] = []
    inputs: list[int] = []
    targets: list[int] = []
    packed_tasks: list[MultipleChoiceTask] = []

    def flush_sequence() -> None:
        nonlocal inputs, targets, packed_tasks
        sequence = finalize_sequence(
            inputs,
            targets,
            packed_tasks,
            seq_len=seq_len,
            eot_id=eot_id,
            doc_len_min_average=doc_len_min_average,
        )
        sequences.append(sequence)
        inputs, targets, packed_tasks = [], [], []

    for item in items:
        query_tokens = item["query_tokens"]
        choices_tokens = item["choices_tokens"]
        choice_char_lengths: list[int] = item["choice_char_lengths"]
        label = item["label"]

        total_task_tokens = sum(len(query_tokens) + len(ct) for ct in choices_tokens)
        if len(inputs) + total_task_tokens >= seq_len:
            flush_sequence()

        spans = []
        for choice_tokens in choices_tokens:
            if len(inputs) > 0:
                inputs.append(eot_id)
                targets.append(eot_id)
            context_plus_choice = query_tokens + choice_tokens
            start = len(inputs) + len(query_tokens) - 1
            end_excluding = start + len(choice_tokens)
            spans.append((start, end_excluding))
            inputs.extend(context_plus_choice[:-1])
            targets.extend(context_plus_choice[1:])

        packed_tasks.append(
            MultipleChoiceTask(spans=spans, label=label, choice_char_lengths=choice_char_lengths)
        )

    if len(inputs) > 0:
        flush_sequence()

    return sequences


def score_sequence(
    model: nn.Module,
    sequence: PackedSequence,
    autocast_context=None,
    normalization: NormalizationType = "acc_norm",
) -> tuple[int, int]:
    """Score a packed sequence and return (n_correct, n_count).

    normalization="acc_norm": character-length normalized (sum of loss / char_length).
        Follows lighteval's LogProbCharNorm and lm-evaluation-harness acc_norm.
    normalization="acc": unnormalized sum of loss (favours shorter choices).
    """
    input_seq = sequence.inputs.unsqueeze(0).to(device="cuda")
    target_seq = sequence.targets.unsqueeze(0).to(device="cuda")

    if autocast_context is None:
        from contextlib import nullcontext

        autocast_context = nullcontext()

    with autocast_context:
        logits = model(input_seq)

    loss_per_token = Fun.cross_entropy(
        input=logits.view(-1, logits.size(-1)).float(),
        target=target_seq.view(-1),
        reduction="none",
    )

    n_correct, n_count = 0, 0
    for task in sequence.tasks:
        if normalization == "acc_norm":
            loss_per_choice = [
                loss_per_token[s:e].sum() / char_len
                for (s, e), char_len in zip(task.spans, task.choice_char_lengths, strict=True)
            ]
        else:
            loss_per_choice = [loss_per_token[s:e].sum() for s, e in task.spans]
        choice_with_lowest_loss = torch.stack(loss_per_choice).argmin().item()
        n_correct += int(choice_with_lowest_loss == task.label)
        n_count += 1

    return n_correct, n_count
