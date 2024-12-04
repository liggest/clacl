from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Iterable, Protocol, Callable, TypedDict
from contextlib import contextmanager
from functools import cached_property
from dataclasses import dataclass

import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from clacl.util import wandb_log, logger

if TYPE_CHECKING:
    from transformers import BatchFeature
    from clacl.task.common import WavMLClassificationTask as TaskBase
    from clacl.task.common import WavMLClassificationTrainer as TrainerBase

@dataclass
class Info:
    total: int = 0
    correct: int = 0
    loss: float = 0

    @property
    def accuracy(self):
        return self.correct / self.total

class BatchData(TypedDict):
    phase: Phase
    trainer: TrainerBase
    inputs: BatchFeature
    targets: torch.Tensor
    info: Info

# maybe not a good idea, but let me try for a bit
class Phase:

    name: Literal["train", "valid", "test"]

    @cached_property
    def print_name(self):
        return self.name.capitalize()

    def __init__(self, one_batch: Callable[[BatchData], Info] | None = None):
        self._one_batch = one_batch

    def epoch_info(self):
        return Info()

    def tqdm(self, loader: DataLoader) -> Iterable[tuple[BatchFeature, torch.LongTensor]]:
        return tqdm(loader, desc=f"Batch [{self.name}]", position=1, leave=False)

    def begin_epoch(self, task: TaskBase, loader: DataLoader, info: Info):
        return info
    
    def end_epoch(self, task: TaskBase, loader: DataLoader, info: Info):
        raise NotImplementedError

    @contextmanager
    def on_epoch(self, task: TaskBase, loader: DataLoader, epoch_info: Callable[[Phase, DataLoader], Info | None] | None = None):
        info = None
        if epoch_info:
            info = epoch_info(self, loader)
        if info is None:
            info = self.epoch_info()
        info = self.begin_epoch(task, loader, info)
        try:
            yield info
        finally:
            self.end_epoch(task, loader, info)

    def begin_batch(self, task: TaskBase):
        raise NotImplementedError
    
    def end_batch(self, task: TaskBase):
        raise NotImplementedError

    @contextmanager
    def on_batch(self, task: TaskBase):
        self.begin_batch(task)
        try:
            yield
        finally:
            self.end_batch(task)

    def one_batch(self, data: BatchData):
        if self._one_batch:
            return self._one_batch(data)
        return data["info"]

    def loss(self, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], logits: torch.Tensor, targets: torch.Tensor, *args, **kw) -> torch.FloatTensor:
        raise NotImplementedError

    def update(self, info: Info, total = 0, correct = 0, loss = 0):
        info.total += total
        info.correct += correct

    def print_result(self, info: Info):
        # print(f"{self.print_name} Acc: ", info.accuracy)
        logger.info(f"{self.print_name} Acc: {info.accuracy}")

    def log_data(self, info: Info, epoch_id: int):
        return {
            f"{self.name}/epoch": epoch_id + 1,
            f"{self.name}/epoch_acc": info.accuracy,
        }

    def summary(self, info: Info, epoch_id: int):
        self.print_result(info)
        wandb_log(self.log_data(info, epoch_id))


class TrainPhase(Phase):
    name = "train"

    @cached_property
    def print_name(self):
        return f"Epoch {self.name.capitalize()}"

    def begin_epoch(self, task, loader, info: Info):
        task.model.train()
        return super().begin_epoch(task, loader, info)

    def begin_batch(self, task):
        task.optimizer.zero_grad()
    
    def loss(self, loss_function, logits, targets, *args, **kw):
        loss: torch.FloatTensor = loss_function(logits, targets, *args, **kw)
        loss.backward()
        return loss

    def update(self, info: Info, total=0, correct=0, loss=0):
        super().update(info, total, correct, loss)
        info.loss += loss
        self.batch_loss = loss

    def end_batch(self, task):
        task.optimizer.step()
        wandb_log({'train/loss': self.batch_loss})
    
    def end_epoch(self, task, loader, info):
        info.loss /= len(loader)
        task.scheduler.step()

    def print_result(self, info: Info):
        # print(f"{self.print_name} Loss: ", info.loss)
        logger.info(f"{self.print_name} Loss: {info.loss}")
        # super().print_result(info)
        Phase.print_result(self, info)  # for ValidPhase

    def log_data(self, info: Info, epoch_id: int):
        # log_data = super().log_data(info, epoch_id)
        log_data = Phase.log_data(self, info, epoch_id)  # for ValidPhase
        log_data[f"{self.name}/epoch_loss"] = info.loss
        return log_data

class TestPhase(Phase):
    name = "test"
    zero = torch.FloatTensor([0])

    def begin_epoch(self, task, loader, info: Info):
        task.model.eval()
        # return super().begin_epoch(task, loader)
        return Phase.begin_epoch(self, task, loader, info)  # for ValidPhase

    def begin_batch(self, task):
        self.no_gard = torch.no_grad()
        self.no_gard.__enter__()

    def loss(self, loss_function, logits, targets, *args, **kw):
        return self.zero

    def end_batch(self, task):
        self.no_gard.__exit__(None, None, None)
        del self.no_gard

    def end_epoch(self, task, loader, info):
        pass

class ValidPhase(Phase):
    name = "valid"
    print_name = TrainPhase.print_name

    begin_epoch = TestPhase.begin_epoch
    
    begin_batch = TestPhase.begin_batch
    
    def loss(self, loss_function, logits, targets, *args, **kw) -> torch.FloatTensor:
        return loss_function(logits, targets, *args, **kw)
    
    def update(self, info: Info, total=0, correct=0, loss=0):
        super().update(info, total, correct, loss)
        info.loss += loss
    
    end_batch = TestPhase.end_batch

    def end_epoch(self, task, loader, info):
        info.loss /= len(loader)

    print_result = TrainPhase.print_result
    log_data = TrainPhase.log_data

class CurrentTaskTestPhase(TestPhase):

    def log_data(self, info: Info, epoch_id: int):
        return {
            f"{self.name}/epoch": epoch_id + 1,
            f"{self.name}/current_acc": info.accuracy,
        }
    
class OtherTaskTestPhase(TestPhase):
    def __init__(self, task_name: str, one_batch: Callable[[BatchData], None] | None = None):
        super().__init__(one_batch)
        self.task_name = task_name

    @cached_property
    def print_name(self):
        return f"Task {self.task_name} {self.name.capitalize()}"

    def log_data(self, info: Info, epoch_id: int):
        return {
            # f"{self.name}/epoch": epoch_id + 1,
            f"{self.name}/{self.task_name}_acc": info.accuracy,
        }

class PPhases(Protocol):
    train: Phase | None
    valid: Phase | None
    test:  Phase | None

@dataclass
class Phases:
    train: TrainPhase | None
    valid: ValidPhase | None
    test:  TestPhase  | None
