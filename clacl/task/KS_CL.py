# Keyword Spotting Continual Learning

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Iterable
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
from contextlib import contextmanager
from itertools import pairwise

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from clacl.model.wavml_cl import AdaWavLMForSequenceClassification
from clacl.data.common import DataPieceCollator as Collator, DataLoaders
from clacl.data.speech_commands import Dataset
from clacl.task.common import WavMLClassificationTask as TaskBase
from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import TaskConfig, _init_config
from clacl.task.IC import LearningRate, Task as ICTask, Trainer as ICTrainer
from clacl.task.KS import Scheduler, Task as KSTask
from clacl.util import get_device, wandb_log

if TYPE_CHECKING:
    from transformers import BatchFeature
    from transformers.modeling_outputs import SequenceClassifierOutput
    from clacl.task.common import Config

class KSCLConfig(TaskConfig):
    name: str = "KS_CL"
    csv_path: Path = Path("data/KS_CL/csv")
    data_path: Path = Path("data/KS/speech_commands_v0.01")
    # test_path: Path = Path("data/KS/speech_commands_test_set_v0.01")

    epochs: int = 10
    batch_size: int = 32

    learning_rate: LearningRate = LearningRate()
    scheduler: Scheduler = Scheduler()

def task_id2name(task_id: int):
    return f"task{task_id}"

class CLTask(TaskBase):

    def __init__(self, tasks: list[KSSubTask]):
        super().__init__()
        assert tasks
        self.tasks = tasks
        self.task_id = 0

    @property
    def current_task(self):
        return self.tasks[self.task_id]

    @property
    def _raw_config(self):
        return self.current_task._raw_config

    @property
    def config(self):
        return self.current_task.config
    
    @property
    def model(self):
        return self.current_task.model
    
    @property
    def optimizer(self):
        return self.current_task.optimizer
    
    @property
    def scheduler(self):
        return self.current_task.scheduler

    def init_model(self):
        # self.current_task.init_model()
        self.current_task._model()
        

    def _trainer(self):
        return CLTrainer(self)

class CLTrainer(TrainerBase):

    if TYPE_CHECKING:
        task: CLTask

    def train(self):
        cl = self.task
        last_task = None
        total_tasks = len(cl.tasks)
        self.learned_averages = []
        for cl.task_id, task in enumerate(cl.tasks):
            print(f"Task: {cl.task_id + 1} / {total_tasks}")
            print(repr(task.classes))
            if last_task:
                task.inherit(last_task)
            task.add_module(task_id2name(cl.task_id))
            task.train()
            self.evaluate_all_for(task)
            last_task = task

    def evaluate_all_for(self, current_task: KSSubTask):
        cl = self.task
        learned_average_accuracy = 0
        for task_id, test_task in enumerate(cl.tasks):
            test_task.model = current_task.model

            if task_id <= cl.task_id:
                test_task.model.set_task(task_id2name(task_id))  # task_id > cl.task_id => use adapter of current task 
            
            test_task._raw_config = current_task._raw_config
            trainer = test_task._trainer()
            trainer.epoch_id = 0
            trainer.device = get_device()
            phase = OtherTaskTestPhase(task_id)
            loader = test_task._data_loaders.test

            test_info = trainer.one_phase(phase, loader)

            if task_id <= cl.task_id:
                learned_average_accuracy += test_info.accuracy
        
        learned_average_accuracy /= (cl.task_id + 1)

        self.learned_averages.append(learned_average_accuracy)
        
        if len(self.learned_averages) > 1:
            backward_transfer = sum(a1 - a0 for a0, a1 in pairwise(self.learned_averages))
            backward_transfer /= (len(self.learned_averages) - 1)
        else:
            backward_transfer = 0
        wandb_log({
            "test/learned_average_acc": learned_average_accuracy,
            "test/backward_transfer": backward_transfer
        })


class KSSubTask(TaskBase):

    if TYPE_CHECKING:
        model: AdaWavLMForSequenceClassification
        scheduler: ExponentialLR

        _raw_config: Config[KSCLConfig]
        config: KSCLConfig

    def __init__(self, classes: list[str]):
        super().__init__()
        self.classes = classes

    @cached_property
    def _raw_config(self):
        return _init_config(KSCLConfig())
    
    extractor = KSTask.extractor

    @property
    def _data_loaders(self):
        data_path = self.config.data_path
        # test_path = self.config.test_path
        csv_path = self.config.csv_path
        # no _silence_, no need to use DatasetWithSilence
        train_dataset = Dataset(data_path, csv_path / "train.csv", self.classes)
        val_dataset = Dataset(data_path, csv_path / "valid.csv", self.classes)
        test_dataset = Dataset(data_path, csv_path / "test.csv", self.classes)

        collator = Collator(self.extractor)
        batch_size = self.config.batch_size

        train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)

        val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)

        test_loader = DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=12)

        # return {'train':train_loader, 'val':val_loader}
        return DataLoaders(train_loader, val_loader, test_loader)

    @property
    def model_config(self):
        # no _unknown_ and _silence_
        return {
            "id2label": Dataset.id2label,
            "label2id": Dataset.label2id,
            "num_labels": len(Dataset.label2id),
            "classifier_proj_size": 256
        }

    # _model = ICTask._model
    def _model(self):
        self.model = AdaWavLMForSequenceClassification.from_pretrained(self.config.pretrained_name, **self.model_config)
        # self.model.add_task("test")
        return self.model
    
    _edit_model = ICTask._edit_model
    _optimizer = ICTask._optimizer

    _scheduler = KSTask._scheduler

    def _trainer(self):
        return Trainer(self)
    
    def inherit(self, pre_task: KSSubTask):
        self.model = pre_task.model
        self._raw_config = pre_task._raw_config
        
        self.model.set_task_grad(freeze=True)  # freeze previous task
        # self._optimizer()
        # self._scheduler()

    def add_module(self, task_name: str):
        self.model.add_task(task_name)
        self._optimizer()
        self._scheduler()


@dataclass
class Info:
    total: int = 0
    correct: int = 0
    loss: float = 0

    @property
    def accuracy(self):
        return self.correct / self.total

# maybe not a good idea, but let me try for a bit
class Phase:

    name: Literal["train", "valid", "test"]

    @cached_property
    def print_name(self):
        return self.name.capitalize()

    def epoch_info(self):
        return Info()

    def tqdm(self, loader: DataLoader) -> Iterable[tuple[BatchFeature, torch.LongTensor]]:
        return tqdm(loader, desc=f"Batch [{self.name}]", position=1, leave=False)

    def begin_epoch(self, task: TaskBase, loader: DataLoader):
        return self.epoch_info()
    
    def end_epoch(self, task: TaskBase, loader: DataLoader, info: Info):
        raise NotImplementedError

    @contextmanager
    def on_epoch(self, task: TaskBase, loader: DataLoader):
        info = self.begin_epoch(task, loader)
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

    def loss(self, loss_function: nn.Module, logits: torch.Tensor, targets: torch.Tensor) -> torch.FloatTensor:
        raise NotImplementedError

    def update(self, info: Info, total = 0, correct = 0, loss = 0):
        info.total += total
        info.correct += correct

    def print_result(self, info: Info):
        print(f"{self.print_name} Acc: ", info.accuracy)

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

    def begin_epoch(self, task, loader):
        task.model.train()
        return super().begin_epoch(task, loader)

    def begin_batch(self, task):
        task.optimizer.zero_grad()
    
    def loss(self, loss_function, logits, targets):
        loss: torch.FloatTensor = loss_function(logits, targets)
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
        print(f"{self.print_name} Loss: ", info.loss)
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

    def begin_epoch(self, task, loader):
        task.model.eval()
        # return super().begin_epoch(task, loader)
        return Phase.begin_epoch(self, task, loader)  # for ValidPhase

    def begin_batch(self, task):
        self.no_gard = torch.no_grad()
        self.no_gard.__enter__()

    def loss(self, loss_function, logits, targets):
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
    
    def loss(self, loss_function, logits, targets) -> torch.FloatTensor:
        return loss_function(logits, targets)
    
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
    def __init__(self, task_id: int) -> None:
        super().__init__()
        self.task_id = task_id

    @cached_property
    def print_name(self):
        return f"Task {self.task_id} {self.name.capitalize()}"

    def log_data(self, info: Info, epoch_id: int):
        return {
            # f"{self.name}/epoch": epoch_id + 1,
            f"{self.name}/task[{self.task_id}]_acc": info.accuracy,
        }

class Trainer(TrainerBase):

    if TYPE_CHECKING:
        task: KSSubTask

    loss_function = ICTrainer.loss_function

    def train(self):
        self.device = get_device()

        self.task.model.to(self.device)
        loaders = self.task._data_loaders

        torch.backends.cudnn.benchmark = True

        train_phase, valid_phase, test_phase = TrainPhase(), ValidPhase(), CurrentTaskTestPhase()

        self.total_epochs = self.task.config.epochs
        for self.epoch_id in tqdm(range(self.total_epochs), desc="Epoch", position=0):
            print(f"Epoch: {self.epoch_id + 1} / {self.total_epochs}")
            self.one_phase(train_phase, loaders.train)
            self.one_phase(valid_phase, loaders.valid)
        self.one_phase(test_phase, loaders.test)

    def one_phase(self, phase: Phase, loader: DataLoader):
        task = self.task
        with phase.on_epoch(task, loader) as info:
            for inputs, targets in phase.tqdm(loader):
                with phase.on_batch(task):
                    inputs = inputs.to(self.device)

                    output: SequenceClassifierOutput = task.model(**inputs)
                    logits = output.logits.cpu()

                    loss = phase.loss(self.loss_function, logits, targets)

                    _, predict = torch.max(logits.data, 1)
                    phase.update(info,
                        total=targets.size(0),
                        correct=(predict == targets).sum().item(),
                        loss=loss.item()
                    )
        phase.summary(info, self.epoch_id)
        return info

    # def train_phase(self, loader: DataLoader):
    #     task = self.task
    #     task.model.train()
    #     # task.model.eval()
    #     epoch_info = Info("train")

    #     for inputs, targets in tqdm(loader, desc="Batch [train]", position=1, leave=False):
    #         inputs: BatchFeature
    #         targets: torch.LongTensor
    #         inputs = inputs.to(self.device)
            
    #         task.optimizer.zero_grad()

    #         output: SequenceClassifierOutput = task.model(**inputs)
    #         logits = output.logits.cpu()
    #         loss: torch.Tensor = self.loss_function(logits, targets)
    #         loss.backward()

    #         task.optimizer.step()
    #         epoch_info.loss += loss.item()
    #         epoch_info.total += targets.size(0)
    #         _, predict = torch.max(logits.data, 1)
    #         epoch_info.correct += (predict == targets).sum().item()
        
    #     epoch_info.loss /= len(loader)

    #     task.scheduler.step()

    #     print("Epoch Train Loss: ", epoch_info.loss)
    #     print("Epoch Train Acc:  ", epoch_info.accuracy)

    #     wandb_log({
    #         "train/epoch": self.epoch_id + 1,
    #         "train/epoch_acc": epoch_info.accuracy,
    #         "train/epoch_loss": epoch_info.loss
    #     })

