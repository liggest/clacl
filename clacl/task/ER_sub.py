from __future__ import annotations
from typing import TYPE_CHECKING, Literal #, Iterable
from functools import cached_property
from pathlib import Path
from dataclasses import dataclass

from pydantic import BaseModel
import torch
import torch.nn as nn
# from torch.utils.data import Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from clacl.data.common import DataPieceCollator as Collator, DataLoaders
from clacl.data.iemocap import Dataset
from clacl.task.common import _init_config
from clacl.task.phase import BatchData, Info, Phases, Phase
from clacl.task.phase import TrainPhase, ValidPhase, CurrentTaskTestPhase, OtherTaskTestPhase
from clacl.task.sub import SubTask, SubTrainer
from clacl.task.sub import SubTaskConfig, DatasetConfig, TaskDataConfig
from clacl.task.sub import edit_model
from clacl.task.IC import LearningRate
from clacl.util import logger

if TYPE_CHECKING:
    from transformers.modeling_outputs import SequenceClassifierOutput

class IEMOCAPConfig(DatasetConfig):
    type: Literal["IEMOCAP"] = "IEMOCAP"
    path: Path = Path("data/ER_sub/iemocap")

class ERSubDataConfig(TaskDataConfig):
    dataset: str = "IEMOCAP"
    csv_path: Path = Path("data/ER_sub/csv")

class Scheduler(BaseModel):
    type: str = "StepLR"
    step_size: int = 10
    gamma: float = 0.1

class ERSubConfig(SubTaskConfig):
    type: Literal["ERSubTask"] = "ERSubTask"
    name: str = "ER_sub"
    
    data: ERSubDataConfig = ERSubDataConfig()

    epochs: int = 20
    batch_size: int = 16
    # k_folds: int = 5

    learning_rate: LearningRate = LearningRate(adapter_to_output=1e-4, adapter_layer_weights=1e-4)
    scheduler: Scheduler = Scheduler()

    @classmethod
    def with_all_fields(cls):
        config = super().with_all_fields()
        config.dataset = { "IEMOCAP": IEMOCAPConfig() }
        return config

class ERSubTask(SubTask[ERSubConfig]):

    @cached_property
    def _raw_config(self):
        # run sub task directly
        return _init_config(ERSubConfig.with_all_fields())
    
    # @cached_property
    # def _k_fold_state(self) -> tuple[int, tuple[Iterable[int], Iterable[int]]] | None:
    #     return None

    # @cached_property
    # def _dataset(self):
    #     data_config = self.dataset[self.config.data.dataset]
    #     data_path = data_config.path  # path for IEMOCAP_full_release
    #     csv_path = self.config.data.csv_path
    #     model_config = self.model.config                    # 5 labels
    #     return Dataset(data_path, csv_path / "data.csv", list(model_config.label2id), model_config)

    # @property
    # def _data_loaders(self):
    #     train_collator = Collator(self.extractor, lthresh=200000)
    #     val_collator = Collator(self.extractor)
    #     batch_size = self.config.batch_size

    #     if self._k_fold_state is None:
    #         return None
    #     k, (train_indices, val_indices) = self._k_fold_state
    #     dataset = self._dataset
    #     train_dataset = Subset(dataset, train_indices)
    #     val_dataset = Subset(dataset, val_indices)

    #     train_loader = DataLoader(train_dataset, collate_fn=train_collator, batch_size=batch_size, shuffle=True, num_workers=12)
    #     val_loader = DataLoader(val_dataset, collate_fn=val_collator, batch_size=batch_size, shuffle=True, num_workers=12)

    #     return DataLoaders(train_loader, val_loader, None)

    @cached_property
    def classes(self):
        return list(self.model.config.label2id)

    @property
    def _data_loaders(self):
        data_config = self.dataset[self.config.data.dataset]
        data_path = data_config.path  # path for IEMOCAP_full_release
        csv_path = self.config.data.csv_path
        
        model_config = self.model.config  # model must exist at this point
        train_dataset = Dataset(data_path, csv_path / "train.csv", self.classes, label_config=model_config)
        val_dataset = Dataset(data_path, csv_path / "valid.csv", self.classes, label_config=model_config)
        test_dataset = Dataset(data_path, csv_path / "test.csv", self.classes, label_config=model_config)

        batch_size = self.config.batch_size
        
        collator = Collator(self.extractor, lthresh=200000)

        train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)
        val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)
        test_loader = DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=12)

        return DataLoaders(train_loader, val_loader, test_loader)

    @property
    def extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
    
    @cached_property
    def model_config(self):
        config = super().model_config
        label2id = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
        config["label2id"] = label2id
        config["id2label"] = {v: k for k, v in label2id.items() if k != "exc"}  # exc is merged with hap
        config["num_labels"] = len(config["id2label"])  # 4 classes
        return config
    
    def _optimizer(self):
        (down_param, 
        layernorm_param, 
        adapter_param, 
        adapter_to_output_layer_weights_param, 
        adapter_to_output_param
        ) = edit_model(self.model)
        learning_rate = self.config.learning_rate
        self.optimizer = Adam([
            {'params': down_param, 'lr': learning_rate.down},
            {'params': layernorm_param, 'lr': learning_rate.layer_norm},
            {'params': adapter_param, 'lr': learning_rate.adapter_ff},
            {'params': adapter_to_output_layer_weights_param, 'lr': learning_rate.adapter_layer_weights},
            {'params': adapter_to_output_param, 'lr': learning_rate.adapter_to_output},
        ])
        return self.optimizer
    
    def _scheduler(self):
        _config = self.config.scheduler
        self.scheduler = StepLR(self.optimizer, step_size=_config.step_size, gamma=_config.gamma)
        return self.scheduler
    
    def _trainer(self):
        return ERSubTrainer(self)

class ERSubTrainer(SubTrainer[ERSubTask]):
    
    @cached_property
    def total_epochs(self):
        return self.task.config.epochs

    @cached_property
    def loss_function(self):
        return nn.CrossEntropyLoss()
    
    @property
    def _phases(self):
        return Phases(
            TrainPhase(self.one_batch), 
            ERSubValidPhase(self.one_batch), 
            ERSubCurrentTestPhase(self.one_batch)
        )

    # def train(self):
    #     self.device = get_device()

    #     self.task.model.to(self.device)
    #     phases = self._phases

    #     config = self.task.config
    #     kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    #     acc = []
    #     weighted_acc = []
    #     for k, (train_indices, val_indices) in enumerate(kf.split(range(len(self.task._dataset)))):
    #         logger.info(f"Fold: {k + 1} / {config.k_folds}")
    #         self.task._k_fold_state = (k, (train_indices, val_indices))
    #         logger.debug(f"Val indices: {val_indices!r}")
    #         loaders = self.task._data_loaders
    #         assert loaders

    #         if k != 0:
    #             self.task._optimizer()  # reset optimizer and scheduler
    #             self.task._scheduler()

    #         valid_acc = []
    #         valid_weighted_acc = []

    #         for self.epoch_id in tqdm(range(self.total_epochs), desc="Epoch", position=0):
    #             logger.info(f"Epoch: {self.epoch_id + 1} / {self.total_epochs}")
    #             self.one_phase(phases.train, loaders.train)
    #             valid_info: ERSubInfo = self.one_phase(phases.valid, loaders.valid)
    #             valid_acc.append(valid_info.accuracy)
    #             valid_weighted_acc.append(valid_info.weighted_accuracy)

    #         logger.info(f"[Fold {k + 1}] Max Valid Acc: {max(valid_acc)}")
    #         logger.info(f"[Fold {k + 1}] Max Valid Weighted Acc: {max(valid_weighted_acc)}")
    #         acc.append(valid_acc[-1])
    #         weighted_acc.append(valid_weighted_acc[-1])
    #     self.task._k_fold_state = None
    #     del self.task._dataset
    #     acc, weighted_acc = torch.tensor(acc), torch.tensor(weighted_acc)
    #     test_info = ERSubTestInfo(accuracy_mean=acc.mean().item(), accuracy_std=acc.std().item(), weighted_accuracy_mean=weighted_acc.mean().item(), weighted_accuracy_std=weighted_acc.std().item())
    #     phases.test.summary(test_info, self.total_epochs * config.k_folds - 1)
    #     # self.one_phase(phases.test, loaders.test)

    def _epoch_info(self, phase: Phase, loader: DataLoader) -> ERSubInfo:
        num_labels = self.task.model.config.num_labels
        return ERSubInfo(
            total_per_class=torch.zeros(num_labels, dtype=torch.long), 
            correct_per_class=torch.zeros(num_labels, dtype=torch.long)
        )
    
    def one_batch(self, data: BatchData):
        inputs, targets, phase, info = data['inputs'], data['targets'], data['phase'], data['info']
        inputs = inputs.to(self.device)
        # targets = targets.to(self.device)

        output: SequenceClassifierOutput = self.task.model(**inputs)
        logits = output.logits.cpu()

        loss = phase.loss(self.loss_function, logits, targets)

        # _, predict = torch.max(logits.data, 1)
        predicts = torch.argmax(logits, dim=-1)
        corrects = (predicts == targets)

        phase.update(info,
            total=targets.size(0),
            correct=corrects.sum().item(),
            loss=loss.item()
        )

        if isinstance(info, ERSubInfo):
            corrects = corrects.long()
            # info.correct_per_class[targets[i]] += corrects[i]
            # info.total_per_class[targets[i]] += 1
            info.correct_per_class = info.correct_per_class.scatter_add(0, targets, corrects)
            info.total_per_class = info.total_per_class.scatter_add(0, targets, torch.ones_like(targets))
        
        return info
    
    @property
    def _evaluate_for_phase(self) -> Phase:
        return ERSubOtherTestPhase(self.task.name_with_id, self.one_batch)

    # def evaluate_for(self, target_task):
    #     # evaluate target_task (current trained) with self.task
    #     self_task = self.task
    #     self_task.inherit(target_task)

    #     if self_task.task_id <= target_task.task_id:
    #         self_task.model.set_task(self_task.name_with_id)
    #         # test task id > trained task id => use adapter of trained task, except head
        
    #     self.device = get_device()
    #     with self_task.ensure_task_head(self.device):
    #         self.epoch_id = 0
    #         phases = self._phases
    #         config = self.task.config
    #         kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    #         acc = []
    #         weighted_acc = []
    #         for k, (train_indices, val_indices) in enumerate(kf.split(range(len(self.task._dataset)))):
    #             logger.info(f"Fold: {k + 1} / {config.k_folds}")
    #             self.task._k_fold_state = (k, (train_indices, val_indices))
    #             loaders = self.task._data_loaders
    #             assert loaders

    #             valid_info: ERSubInfo = self.one_phase(phases.valid, loaders.valid)
    #             acc.append(valid_info.accuracy)
    #             weighted_acc.append(valid_info.weighted_accuracy)

    #     self.task._k_fold_state = None
    #     del self.task._dataset
    #     acc, weighted_acc = torch.tensor(acc), torch.tensor(weighted_acc)
    #     test_info = ERSubTestInfo(accuracy_mean=acc.mean().item(), accuracy_std=acc.std().item(), weighted_accuracy_mean=weighted_acc.mean().item(), weighted_accuracy_std=weighted_acc.std().item())
    #     ERSubOtherTestPhase(self_task.name_with_id, self.one_batch).summary(test_info, self.epoch_id)
    #     return test_info

@dataclass
class ERSubInfo(Info):
    total_per_class: torch.Tensor = torch.zeros(0, dtype=torch.long)
    correct_per_class: torch.Tensor = torch.zeros(0, dtype=torch.long)

    @property
    def weighted_accuracy(self):
        return (self.correct_per_class / self.total_per_class).mean()

class ERSubValidPhase(ValidPhase):

    def log_data(self, info: ERSubInfo, epoch_id: int):
        log_data = ValidPhase.log_data(self, info, epoch_id)
        log_data[f"{self.name}/epoch_weighted_acc"] = info.weighted_accuracy
        return log_data

class ERSubCurrentTestPhase(CurrentTaskTestPhase):

    def print_result(self, info: ERSubInfo):
        logger.info(f"{self.print_name} Acc: {info.weighted_accuracy}")


    def log_data(self, info: ERSubInfo, epoch_id: int):
        return {
            f"{self.name}/epoch": epoch_id + 1,
            f"{self.name}/current_acc": info.weighted_accuracy,
        }

class ERSubOtherTestPhase(OtherTaskTestPhase):

    print_result = ERSubCurrentTestPhase.print_result

    def log_data(self, info: ERSubInfo, epoch_id: int):
        return {
            f"{self.name}/{self.task_name}_acc": info.weighted_accuracy,
        }
