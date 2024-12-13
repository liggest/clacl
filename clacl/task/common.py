from __future__ import annotations

from typing import TypeVar, TYPE_CHECKING, Generic
from functools import cached_property
from pathlib import Path

from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from pydantic import BaseModel

from clacl.model.wavlm import AdaWavLMForSequenceClassification
from clacl.data.common import PDataLoaders
from clacl.config import file_config_base

class TaskConfig(BaseModel):
    name: str = "Task"
    
    pretrained_name: str = "microsoft/wavlm-base-plus"
    optimizer: str = "Adam"

    seed: int | None = 42

TTaskConfig = TypeVar("TTaskConfig", bound=TaskConfig)
TTaskInConfig = TypeVar("TTaskInConfig", bound=TaskConfig)

class WandbConfig(BaseModel):
    enable: bool = True
    project: str = "clacl"
    name: str | None = "test"
    notes: str | None = "Test run"
    tags: list[str] | None = None
    group: str | None = None

def _init_config(task_config: TTaskConfig) -> Config[TTaskConfig]:
    if TYPE_CHECKING:
        global Config

    task_name = task_config.name
    task_config_type = type(task_config)

    class Config(file_config_base(path=Path(f"data/{task_name}/config.toml")), Generic[TTaskInConfig]):
        if TYPE_CHECKING:
            task: TTaskInConfig = task_config
        else:
            task: task_config_type = task_config
        wandb: WandbConfig = WandbConfig(project=f"clacl_{task_name}")

    return Config[task_config_type]()

class WavMLClassificationTask:

    model: AdaWavLMForSequenceClassification
    optimizer: Adam
    scheduler: LRScheduler

    @cached_property
    def _raw_config(self):
        return _init_config(TaskConfig())

    @cached_property
    def config(self):
        return self._raw_config.task

    @property
    def _data_loaders(self) -> PDataLoaders:
        raise NotImplementedError

    def _model(self) -> nn.Module:
        raise NotImplementedError
    
    def _optimizer(self) -> Optimizer:
        raise NotImplementedError
    
    def _scheduler(self) -> LRScheduler:
        raise NotImplementedError
    
    def init_model(self):
        self._model()
        self._optimizer()
        self._scheduler()

    def _trainer(self) -> WavMLClassificationTrainer:
        return WavMLClassificationTrainer(self)

    def train(self):
        trainer = self._trainer()
        return trainer.train()
    
    def evaluate(self):
        raise NotImplementedError

# TTask = TypeVar("TTask", bound=WavMLClassificationTask)

class WavMLClassificationTrainer:

    def __init__(self, task: WavMLClassificationTask):
        self.task = task

    def train(self):
        raise NotImplementedError

