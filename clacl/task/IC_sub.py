from typing import TYPE_CHECKING, Literal
from functools import cached_property
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from clacl.data.common import DataPieceCollator as Collator, DataLoaders
from clacl.data.fluent_commands import Dataset
from clacl.task.common import _init_config
from clacl.task.phase import BatchData
from clacl.task.sub import SubTask, SubTrainer
from clacl.task.sub import SubTaskConfig, DatasetConfig, TaskDataConfig
from clacl.task.sub import edit_model
from clacl.task.IC import LearningRate, Scheduler

if TYPE_CHECKING:
    from transformers.modeling_outputs import SequenceClassifierOutput

class FluentSpeechCommandsConfig(DatasetConfig):
    type: Literal["FluentSpeechCommands"] = "FluentSpeechCommands"
    path: Path = Path("data/IC/fluent_speech_commands_dataset")

class ICSubDataConfig(TaskDataConfig):
    dataset: str = "FluentSpeechCommands"
    csv_path: Path = Path("data/IC_sub/csv")

class ICSubConfig(SubTaskConfig):
    type: Literal["ICSubTask"] = "ICSubTask"
    name: str = "IC_sub"
    
    data: ICSubDataConfig = ICSubDataConfig()
    # classes: list[str] = CLASSES

    epochs: int = 7
    batch_size: int = 16

    learning_rate: LearningRate = LearningRate()
    scheduler: Scheduler = Scheduler()

    @classmethod
    def with_all_fields(cls):
        config = super().with_all_fields()
        config.dataset = { "FluentSpeechCommands": FluentSpeechCommandsConfig() }
        return config

class ICSubTask(SubTask[ICSubConfig]):
    
    @cached_property
    def _raw_config(self):
        # run sub task directly
        return _init_config(ICSubConfig.with_all_fields())
    
    @property
    def extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-ic")
    
    @property
    def _data_loaders(self):
        data_config = self.dataset[self.config.data.dataset]
        data_path = data_config.path  # path for fluent_speech_commands_dataset
        csv_path = self.config.data.csv_path
        
        model_config = self.model.config  # model must exist at this point
        train_dataset = Dataset(model_config, data_path, csv_path / "train.csv")
        val_dataset = Dataset(model_config, data_path, csv_path / "valid.csv")
        test_dataset = Dataset(model_config, data_path, csv_path / "test.csv")

        batch_size = self.config.batch_size
        
        collator = Collator(self.extractor)

        train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)
        val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=12)
        test_loader = DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=12)

        return DataLoaders(train_loader, val_loader, test_loader)
    
    @cached_property
    def model_config(self):
        config = super().model_config
        config["id2label"] = Dataset.id2label
        config["label2id"] = Dataset.label2id
        config["num_labels"] = len(Dataset.label2id)
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
        def lr_lambda(epoch: int):
            return self.config.scheduler.step[epoch]

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        return self.scheduler
    
    def _trainer(self):
        return ICSubTrainer(self)

class ICSubTrainer(SubTrainer[ICSubTask]):

    @cached_property
    def total_epochs(self):
        return self.task.config.epochs

    @cached_property
    def loss_function(self):
        return nn.CrossEntropyLoss()

    def _cal_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                  logits_act: torch.Tensor, logits_obj: torch.Tensor, logits_loc: torch.Tensor,
                  act_ids: torch.Tensor, obj_ids: torch.Tensor, loc_ids: torch.Tensor
                ):
        loss_act = self.loss_function(logits_act, act_ids)
        loss_obj = self.loss_function(logits_obj, obj_ids - Dataset.ACT_STOP)
        loss_loc = self.loss_function(logits_loc, loc_ids - Dataset.OBJ_STOP)
        loss: torch.Tensor = loss_act + loss_obj + loss_loc
        return loss
    
    def one_batch(self, data: BatchData):
        inputs, targets, phase, info  = data['inputs'], data['targets'], data['phase'], data['info']
        batch_size: int = inputs['input_values'].size(0)

        inputs = inputs.to(self.device)
        output: SequenceClassifierOutput = self.task.model(**inputs)
        logits = output.logits.cpu()

        act_ids = targets[:,0]
        obj_ids = targets[:,1]
        loc_ids = targets[:,2]
        logits_act = logits[:, :Dataset.ACT_STOP]
        logits_obj = logits[:, Dataset.ACT_STOP: Dataset.OBJ_STOP]
        logits_loc = logits[:, Dataset.OBJ_STOP: Dataset.LOC_STOP]
        act_pred_ids = logits_act.argmax(dim=-1)
        obj_pred_ids = logits_obj.argmax(dim=-1)
        loc_pred_ids = logits_loc.argmax(dim=-1)

        act_corrects = act_pred_ids.squeeze().eq(act_ids)
        obj_corrects = obj_pred_ids.squeeze().eq(obj_ids - Dataset.ACT_STOP)
        loc_corrects = loc_pred_ids.squeeze().eq(loc_ids - Dataset.OBJ_STOP)
        corrects = torch.stack([act_corrects, obj_corrects, loc_corrects], dim=1)

        loss = phase.loss(
            self._cal_loss, logits, targets,
            logits_act, logits_obj, logits_loc,
            act_ids, obj_ids, loc_ids
        )

        phase.update(info,
            total=batch_size,
            correct=corrects.prod(1).float().sum().item(),
            loss=loss.item()
        )
        return info

