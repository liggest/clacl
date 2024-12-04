from typing import TYPE_CHECKING, Literal
from functools import cached_property
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
# from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from pydantic import BaseModel

from clacl.data.common import DataPieceCollator as Collator, DataLoaders, seed_worker
from clacl.data.asvspoof_2019 import CLASSES, Dataset
from clacl.task.common import _init_config
from clacl.task.phase import BatchData
from clacl.task.sub import SubTask, SubTrainer
from clacl.task.sub import SubTaskConfig, DatasetConfig, TaskDataConfig
from clacl.task.sub import edit_model
from clacl.task.IC import LearningRate
# from clacl.task.KS import Scheduler
from clacl.util import logger

if TYPE_CHECKING:
    from transformers.modeling_outputs import SequenceClassifierOutput

class ASVspoof2019Config(DatasetConfig):
    type: Literal["ASVspoof2019"] = "ASVspoof2019"
    path: Path = Path("data/FSD_sub/ASVspoof2019/LA")

class FSDSubDataConfig(TaskDataConfig):
    dataset: str = "ASVspoof2019"
    csv_path: Path = Path("data/FSD_sub/csv")

class Scheduler(BaseModel):
    type: str = "CosineAnnealingLR"
    T_max: int = 4
    eta_min: float = 1e-6

class FSDSubConfig(SubTaskConfig):
    type: Literal["FSDSubTask"] = "FSDSubTask"
    name: str = "FSD_sub"
    
    data: FSDSubDataConfig = FSDSubDataConfig()
    classes: list[str] = CLASSES

    epochs: int = 8
    batch_size: int = 16

    learning_rate: LearningRate = LearningRate(down=1e-4)
    scheduler: Scheduler = Scheduler()

    @classmethod
    def with_all_fields(cls):
        config = super().with_all_fields()
        config.dataset = { "ASVspoof2019": ASVspoof2019Config() }
        return config

class FSDSubTask(SubTask[FSDSubConfig]):

    @cached_property
    def _raw_config(self):
        # run sub task directly
        return _init_config(FSDSubConfig.with_all_fields())

    @property
    def classes(self):
        return self.config.classes
    
    @property
    def extractor(self):
        # use the same setting to init Wav2Vec2FeatureExtractor
        return Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-ks")
    
    @property
    def _data_loaders(self):
        data_config = self.dataset[self.config.data.dataset]
        data_path = data_config.path  # path for LA
        csv_path = self.config.data.csv_path
        
        model_config = self.model.config  # model must exist at this point
        train_dataset = Dataset(data_path, csv_path / "train.csv", self.classes, label_config=model_config)
        val_dataset = Dataset(data_path, csv_path / "valid.csv", self.classes, label_config=model_config)
        test_dataset = Dataset(data_path, csv_path / "test.csv", self.classes, label_config=model_config)

        batch_size = self.config.batch_size
        
        collator = Collator(self.extractor)

        g = torch.Generator()
        g.manual_seed(self.config.seed)
        train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g)
        # train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=8)
        val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g)
        # val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=seed_worker, generator=g)

        return DataLoaders(train_loader, val_loader, test_loader)
    
    @cached_property
    def model_config(self):
        config = super().model_config
        label2id = {label: i for i, label in enumerate(self.config.classes)}
        id2label = {i: label for label, i in label2id.items()}
        config["id2label"] = id2label
        config["label2id"] = label2id
        config["num_labels"] = len(label2id)
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
    
    # def _scheduler(self):
    #     self.scheduler = ExponentialLR(self.optimizer, self.config.scheduler.gamma)
    #     return self.scheduler
    
    def _scheduler(self):
        config = self.config.scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, config.T_max, config.eta_min)
        return self.scheduler
    
    def _trainer(self):
        return FSDSubTrainer(self)

class FSDSubTrainer(SubTrainer[FSDSubTask]):

    @cached_property
    def loss_function(self):
        test_set: Dataset  = self.task._data_loaders.test.dataset
        label_weights = test_set.label_weights
        label2id = self.task.model.config.label2id
        weights = torch.tensor([label_weights[label] for label in label2id], dtype=torch.float32)
        logger.info(f"CrossEntropy weight for {list(label2id)!r}: {weights}")
        return nn.CrossEntropyLoss(weights)

    @cached_property
    def total_epochs(self):
        return self.task.config.epochs
    
    def one_batch(self, data: BatchData):
        inputs, targets, phase, info  = data['inputs'], data['targets'], data['phase'], data['info']
        inputs = inputs.to(self.device)

        output: SequenceClassifierOutput = self.task.model(**inputs)
        logits = output.logits.cpu()

        loss = phase.loss(self.loss_function, logits, targets)

        # torch.nn.utils.clip_grad_norm_(self.task.model.parameters(), max_norm=1.0)  # gradient clipping

        _, predict = torch.max(logits.data, 1)
        phase.update(info,
            total=targets.size(0),
            correct=(predict == targets).sum().item(),
            loss=loss.item()
        )
        return info
