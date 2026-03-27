from typing import Any

import torch
import torch.distributed as dist


class AverageMeter:
    """
    A class for working with average meters.

    Support for distributed training with `torch.distributed`.
    """

    def __init__(
        self,
        name: str = "",
        sum: int | float = 0.0,
        count: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.name = name
        self.sum = sum
        self.count = count
        self.device = device

    @property
    def average(self) -> float:
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return 0.0

    def update(self, value: int | float, nums: int = 1) -> None:
        self.sum += value * nums
        self.count += nums

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def reduce(self, dst: int) -> None:
        """Perform an in-place reduce."""
        meters_to_reduce = torch.tensor(
            [self.sum, self.count], dtype=torch.float32, device=self.device
        )
        # only `Tensor` of process with rank `dst` will be modified in-place,
        # `Tensor` of other processes will remain the same
        dist.reduce(meters_to_reduce, dst=dst, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def all_reduce(self) -> None:
        """Perform an in-place all reduce."""
        meters_to_reduce = torch.tensor(
            [self.sum, self.count], dtype=torch.float32, device=self.device
        )
        dist.all_reduce(meters_to_reduce, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def gather_object(
        self, dst: int, world_size: int, is_master: bool
    ) -> list[dict[str, Any]] | None:
        output = [None for _ in range(world_size)] if is_master else None
        object_dict = self.to_dict()
        dist.gather_object(object_dict, output, dst)
        assert output is not None if is_master else output is None
        return output  # pyright: ignore[reportReturnType]

    def all_gather_object(self, world_size: int) -> list[dict[str, Any]]:
        output = [None for _ in range(world_size)]
        object_dict = self.to_dict()
        dist.all_gather_object(output, object_dict)
        return output  # pyright: ignore[reportReturnType]

    def __repr__(self) -> str:
        str_repr = f"{self.__class__.__name__}("
        if self.name:
            str_repr += f"name={self.name}, "
        str_repr += f"average={self.average}, sum={self.sum}, count={self.count}"
        if self.device is not None:
            str_repr += f", device={self.device}"
        str_repr += ")"
        return str_repr

    def to_dict(self) -> dict[str, Any]:
        return vars(self)
