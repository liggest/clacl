from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from functools import cached_property
import random
from itertools import pairwise
from enum import Enum

from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
# from tqdm import tqdm

from clacl.model.wavlm_cl import AdapterState
from clacl.data.common import DataPieceCollator as Collator, DataLoaders
from clacl.data.speech_commands import Dataset as SCDataset, CLASSES as SC_CLASSES

from clacl.task.common import WavMLClassificationTask as TaskBase
from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import TaskConfig, _init_config
from clacl.task.IC import LearningRate, Task as ICTask, Trainer as ICTrainer
from clacl.task.KS import Scheduler, Task as KSTask
from clacl.task.KS_CL import KSSubTask as KS_CLSubTask, Trainer as KS_CLSubTrainer
from clacl.task.KS_CL import CLTask as KS_CLTask, task_id2name
# from clacl.task.KS_CL import CLTrainer as KS_CLTrainer
from clacl.task.KS_CL import OtherTaskTestPhase
from clacl.util import get_device, wandb_log

if TYPE_CHECKING:
    from clacl.task.KS_CL import Phase
    from transformers.modeling_outputs import SequenceClassifierOutput


class CILState(str, Enum):
    No = "no"
    AverageAdapter = "average_adapter"
    MaximizeLogit = "max_logit"
    ClassifyTask = "classify_task"

class DataConfig(BaseModel):
    csv_path: Path = Path("data/KS_CIL/csv")
    sc_path: Path = Path("data/KS/speech_commands_v0.01")
    # grabo_path: Path = Path("data/KS_CIL/grabo/grabo")

class ModelConfig(BaseModel):
    e_adapter: AdapterState = AdapterState.CL
    l_adapter: AdapterState = AdapterState.CL
    head: AdapterState = AdapterState.CL

class SubTaskConfig(BaseModel):
    classes: list[str] = SC_CLASSES
    n_first: int = 15
    n_rest: int = 3
    
    cil_state: CILState = CILState.AverageAdapter

class KSCILConfig(TaskConfig):
    name: str = "KS_CIL"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    task: SubTaskConfig = SubTaskConfig()

    epochs: int = 10
    batch_size: int = 32

    learning_rate: LearningRate = LearningRate()
    scheduler: Scheduler = Scheduler()

class KSSubTask(TaskBase):

    if TYPE_CHECKING:
        from torch.optim.lr_scheduler import ExponentialLR

        from clacl.model.wavlm_cl import AdaWavLMForSequenceClassification
        from clacl.task.common import Config
        
        model: AdaWavLMForSequenceClassification
        scheduler: ExponentialLR

        _raw_config: Config[KSCILConfig]
        config: KSCILConfig

    task_id: int

    def __init__(self, classes: list[str], _raw_config: Config[KSCILConfig] | None = None):
        super().__init__()
        self.classes = classes
        if _raw_config:
            self._raw_config = _raw_config

    # def __init__(self, n_classes: int = 0):
    #     super().__init__()
    #     self.n_classes = n_classes

    # @cached_property
    # def classes(self) -> list[str]:
    #     raise ValueError(f"The class list of {self.__class__.__name__}(n_classes = {self.n_classes}) is not initialized")

    # @cached_property
    # def _raw_config(self):
    #     return _init_config(KSCILConfig())
    
    extractor = KSTask.extractor

    @property
    def _data_loaders(self):
        data_path = self.config.data.sc_path
        csv_path = self.config.data.csv_path
        # no _silence_, no need to use DatasetWithSilence
        train_dataset = SCDataset(data_path, csv_path / "train.csv", self.classes)
        val_dataset = SCDataset(data_path, csv_path / "valid.csv", self.classes)
        test_dataset = SCDataset(data_path, csv_path / "test.csv", self.classes)

        collator = Collator(self.extractor)
        batch_size = self.config.batch_size

        train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)

        val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)

        test_loader = DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=12)

        return DataLoaders(train_loader, val_loader, test_loader)

    @property
    def model_config(self):
        label2id = {label: i for i, label in enumerate(self.config.task.classes)}
        id2label = {str(i): label for label, i in label2id.items()}
        return {
            "id2label": id2label, # Dataset.id2label,
            "label2id": label2id, # Dataset.label2id,
            "num_labels": len(label2id), # len(Dataset.label2id),
            "classifier_proj_size": 256,
            "e_adapter_state": self.config.model.e_adapter,
            "l_adapter_state": self.config.model.l_adapter,
            "head_state": self.config.model.head,
        }

    _model = KS_CLSubTask._model
    
    _edit_model = ICTask._edit_model
    _optimizer = ICTask._optimizer

    _scheduler = KSTask._scheduler

    def _trainer(self):
        return Trainer(self)
    
    def inherit(self, pre_task: KSSubTask):
        self.model = pre_task.model
        # self._raw_config = pre_task._raw_config
        
        # self.model.set_task_grad(freeze=True)  # freeze previous task

    def add_module(self, task_id: int):
        self.model.add_task(task_id2name(task_id))
        self._optimizer()
        self._scheduler()

    def evaluate_for(self, target_task: KSSubTask):
        trainer = self._trainer()
        return trainer.evaluate_for(target_task)

class Trainer(TrainerBase):

    if TYPE_CHECKING:
        task: KSSubTask

    loss_function = ICTrainer.loss_function

    train = KS_CLSubTrainer.train

    one_phase = KS_CLSubTrainer.one_phase

    def evaluate_for(self, target_task: KSSubTask):
        task_config = target_task.config.task
        self_task = self.task
        self_task.inherit(target_task)

        if task_config.cil_state == CILState.No:
            if self_task.task_id <= target_task.task_id:
                self_task.model.set_task(task_id2name(self_task.task_id))  
                # task_id > cl.task_id => use adapter of current task
        elif task_config.cil_state == CILState.AverageAdapter:
            print("]] before average:", self_task.model.classifier.current_task)
            self_task.model.set_average_task()
            print("]] after average:", self_task.model.classifier.current_task)


        self.epoch_id = 0
        self.device = get_device()
        phase = OtherTaskTestPhase(self_task.task_id)
        loader = self_task._data_loaders.test

        if task_config.cil_state == CILState.MaximizeLogit:
            test_info = self.one_phase_max_logit(phase, loader)
        else:
            test_info = self.one_phase(phase, loader)

        return test_info
    
    def one_phase_max_logit(self, phase: Phase, loader: DataLoader):
        task = self.task
        with phase.on_epoch(task, loader) as info:
            for inputs, targets in phase.tqdm(loader):
                with phase.on_batch(task):
                    inputs = inputs.to(self.device)

                    max_logits_task = ""
                    max_logits = None
                    max_logits_value = 0

                    for task_name in task.model._task_names_gen():
                        print(f"Under {task_name} model")
                        task.model.set_task(task_name)

                        output: SequenceClassifierOutput = task.model(**inputs)
                        logits = output.logits.cpu()

                        if (logits_max := logits.max().item()) > max_logits_value:
                            max_logits_task = task_name
                            max_logits = logits
                            max_logits_value = logits_max
                            print(f"{max_logits_task = }, {max_logits_value = }")

                    print(f"Final {max_logits_task = }, {max_logits_value = }")

                    # loss = phase.loss(self.loss_function, logits, targets)
                    loss = phase.loss(self.loss_function, max_logits, targets)

                    # _, predict = torch.max(logits.data, 1)
                    _, predict = torch.max(max_logits.data, 1)
                    phase.update(info,
                        total=targets.size(0),
                        correct=(predict == targets).sum().item(),
                        loss=loss.item()
                    )
        phase.summary(info, self.epoch_id)
        return info

class CLTask(TaskBase):

    if TYPE_CHECKING:
        from clacl.task.common import Config

        _raw_config: Config[KSCILConfig]
        config: KSCILConfig

    SubTaskCls = KSSubTask

    def __init__(self):
        super().__init__()
        # self.init_tasks()

    @cached_property
    def _raw_config(self):
        return _init_config(KSCILConfig())

    current_task = KS_CLTask.current_task
    model = KS_CLTask.model
    optimizer = KS_CLTask.optimizer
    scheduler = KS_CLTask.optimizer

    def init_tasks(self):
        task_config = self.config.task
        classes = task_config.classes.copy()
        random.shuffle(classes)
        print("Shuffled classes:", repr(classes))
        self.tasks = [*self._task_gen(classes, task_config)]
        assert self.tasks
        self.classes = classes
        self.task_id = 0

    def _task_gen(self, classes: list[str], config: SubTaskConfig):
        n = config.n_first
        while len(classes) >= n:
            task_classes, classes = classes[:n], classes[n:]
            yield self.SubTaskCls(task_classes, self._raw_config)
            # use CLTask's config
            n = config.n_rest
        
    # init_model = KS_CLTask.init_model
    def init_model(self):
        self.init_tasks()
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
            print(repr(task.classes), f"({len(task.classes)} classes)")
            
            task.task_id = cl.task_id
            if last_task:
                task.inherit(last_task)
            task.add_module(cl.task_id)
            print("]] before train:", task.model.classifier.current_task)
            
            task.train()

            task.model.set_task_grad(freeze=True, f=lambda adapter: adapter.state != AdapterState.TuneAll)

            self.evaluate_all_for(task)

            task.model.set_task(task_id2name(task.task_id))
            print("]] after train:", task.model.classifier.current_task)
            last_task = task

    def evaluate_all_for(self, current_task: KSSubTask):
        cl = self.task
        learned_average_accuracy = 0
        for task_id, test_task in enumerate(cl.tasks):
            test_task.task_id = task_id
            test_info = test_task.evaluate_for(current_task)

            if test_task.task_id <= current_task.task_id:
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
