# Code partially derived from https://github.com/sinhat98/adapter-wavlm/blob/main/IC/train.py and https://github.com/sinhat98/adapter-wavlm/blob/main/IC/utils.py

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from pathlib import Path
from functools import cached_property

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pydantic import BaseModel
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from tqdm import tqdm

from clacl.model.wavml import AdaWavLMForSequenceClassification
from clacl.task.common import WavMLClassificationTask as TaskBase
from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import TaskConfig, _init_config
from clacl.data.common import DataPieceCollator as Collator, DataLoaders
from clacl.data.fluent_commands import Dataset
from clacl.util import get_device, wandb_log

if TYPE_CHECKING:
    from transformers import BatchFeature
    from transformers.modeling_outputs import SequenceClassifierOutput
    from clacl.task.common import Config

class LearningRate(BaseModel):
    down: float = 5e-4
    adapter_to_output: float = 1e-5
    adapter_layer_weights: float = 1e-5
    adapter_ff: float = 1e-5
    layer_norm: float = 1e-5

class Scheduler(BaseModel):
    type: str = "LambdaLR"
    step: list[float] = [0.1, 0.5, 0.7, 1.0, 0.5, 0.3, 0.1, 0]

class ICConfig(TaskConfig):
    name: str = "IC"

    csv_path: Path = Path("data/IC/csv")
    data_path: Path = Path("data/IC/fsc")

    # pretrained_name: str = "microsoft/wavlm-base-plus"
    epochs: int = 7
    batch_size: int = 16
    # optimizer: str = "Adam"
    learning_rate: LearningRate = LearningRate()
    scheduler: Scheduler = Scheduler()

# def _init_config():
#     class Config(file_config_base("data/IC/config.toml")):
#         IC: ICConfig = ICConfig()
#         wandb: WandbConfig = WandbConfig()

#     return Config()

class Task(TaskBase):


    if TYPE_CHECKING:
        scheduler: LambdaLR
        
        _raw_config: Config[ICConfig]
        config: ICConfig

    @cached_property
    def _raw_config(self):
        return _init_config(ICConfig())

    @cached_property
    def pre_config(self):
        return Wav2Vec2Config.from_pretrained("superb/wav2vec2-large-superb-ic")

    @cached_property
    def extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-ic")

    @property
    def _data_loaders(self):
        data_path = self.config.data_path
        csv_path = self.config.csv_path
        train_dataset = Dataset(self.pre_config, data_path, csv_path / "train.csv")
        val_dataset = Dataset(self.pre_config, data_path, csv_path / "valid.csv")
        test_dataset = Dataset(self.pre_config, data_path, csv_path / "test.csv")

        collator = Collator(self.extractor)

        batch_size = self.config.batch_size
        train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=8)

        # return {'train':train_loader, 'val':val_loader, 'test':test_loader}
        return DataLoaders(train_loader, val_loader, test_loader)

    @property
    def model_config(self):
        return {
            "id2label": self.pre_config.id2label,
            "label2id": self.pre_config.label2id,
            "num_labels": len(self.pre_config.id2label),
            "classifier_proj_size": 256
        }

    def _model(self):
        self.model = AdaWavLMForSequenceClassification.from_pretrained(self.config.pretrained_name, **self.model_config)
        return self.model

    def _edit_model(self):
        down_param = []
        # encoder_param = []
        layernorm_param = []
        # layerweight_param = []
        adapter_param = []
        adapter_to_output_param = []
        adapter_to_output_layer_weights_param=[]
        
        frozen, tuned = 0, 0
        
        self.layer_names = [f'layers.{k}' for k in range(0,12)]
        for name, param in self.model.named_parameters():
            flag = any(idx in name for idx in self.layer_names)
            is_not_frozen = param.requires_grad
            
            if is_not_frozen and ('projector' in name or 'classifier' in name):
                print('down: ', name)
                tuned += param.numel()
                down_param.append(param)
            elif is_not_frozen and 'adapter_to_output_layer_weights' in name:
                print('adapter_to_output_layer_weights: ', name)
                tuned += param.numel()
                adapter_to_output_layer_weights_param.append(param)
            # elif ('encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder):
            elif is_not_frozen and ('encoder.layers' in name and 'layer_norm' in name and flag):
                print('layer_norm: ', name)
                tuned += param.numel()
                layernorm_param.append(param)
            elif is_not_frozen and 'adapter_to_output' in name:
                print('adapter_output: ', name)
                tuned += param.numel()
                adapter_to_output_param.append(param)
            elif is_not_frozen and 'adapter_layer' in name:
                print('adapter_layer: ', name)
                tuned += param.numel()
                adapter_param.append(param)
            # elif 'encoder.layers' in name and flag and args.train_encoder:
            #     print('encoder: ', name)
            #     pcount += param.numel()
            #     encoder_param.append(param)
            # elif 'layer_weights' in name and args.weighted_sum:
            #     print('layer_weight: ', name)
            #     pcount+=param.numel()
            #     layerweight_param.append(param)
            else:
                print('frozen: ', name)
                frozen += param.numel()
                param.requires_grad = False
        
        total = tuned + frozen
        print(f"num of tuned params: {tuned} / {total} ({tuned / total * 100:.2f}%)")
        return down_param, layernorm_param, adapter_param, adapter_to_output_layer_weights_param, adapter_to_output_param

    def _optimizer(self):
        (down_param, 
        layernorm_param, 
        adapter_param, 
        adapter_to_output_layer_weights_param, 
        adapter_to_output_param
        ) = self._edit_model()
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
        return Trainer(self)

    def weight_result(self):
        weight = torch.nn.functional.softmax(self.model.wavlm.encoder.adapter_to_output_layer_weights.detach().cpu(), dim=0).numpy()
        return {k: weight[i] for i, k in enumerate(self.layer_names)}

class Trainer(TrainerBase):

    @cached_property
    def loss_function(self):
        return nn.CrossEntropyLoss()

    if TYPE_CHECKING:
        task: Task

    def __init__(self, task: Task):
        super().__init__(task)
        self.act_idx, self.obj_idx, self.loc_idx = 6, 20, 24

    def train(self):
        self.device = get_device()

        self.task.model.to(self.device)
        loaders = self.task._data_loaders

        torch.backends.cudnn.benchmark = True

        self.total_epochs = self.task.config.epochs
        for epoch_id in tqdm(range(self.total_epochs), desc="Epoch", position=0):
            print(f"Epoch: {epoch_id + 1} / {self.total_epochs}")
            self.epoch_id = epoch_id
            for phase, loader in {"train": loaders.train, "val": loaders.valid}.items():
                self.one_phase(phase, loader)
        self.one_phase("test", loaders.test)

    def one_phase(self, phase: Literal["train", "val", "test"] , loader: DataLoader):
        task = self.task
        is_train = phase == "train"
        if is_train:
            task.model.train()
            self.epoch_loss: float = 0.0
        else:
            task.model.eval()

        phase_corrects = 0
        # data_count = 0

        for inputs, targets in tqdm(loader, desc=f"Batch [{phase}]", position=1, leave=False):
            inputs: BatchFeature
            targets: torch.LongTensor
            batch_size: int = inputs['input_values'].size(0)
            # data_count += batch_size

            task.optimizer.zero_grad()

            inputs = inputs.to(self.device)
            output: SequenceClassifierOutput = task.model(**inputs)
            logits = output.logits.cpu()

            act_ids = targets[:,0]
            obj_ids = targets[:,1]
            loc_ids = targets[:,2]
            logits_act = logits[:, :self.act_idx]
            logits_obj = logits[:, self.act_idx: self.obj_idx]
            logits_loc = logits[:, self.obj_idx: self.loc_idx]
            act_pred_ids = logits_act.argmax(dim=-1)
            obj_pred_ids = logits_obj.argmax(dim=-1)
            loc_pred_ids = logits_loc.argmax(dim=-1)
            # preds = torch.stack([act_pred_ids, obj_pred_ids, loc_pred_ids], dim=1)

            act_corrects = act_pred_ids.squeeze().eq(act_ids)
            obj_corrects = obj_pred_ids.squeeze().eq(obj_ids - self.act_idx)
            loc_corrects = loc_pred_ids.squeeze().eq(loc_ids - self.obj_idx)
            corrects = torch.stack([act_corrects, obj_corrects, loc_corrects], dim=1)

            phase_corrects += corrects.prod(1).float().sum().item()

            if is_train:
                with torch.set_grad_enabled(True):
                    loss_act = self.loss_function(logits_act, act_ids)
                    loss_obj = self.loss_function(logits_obj, obj_ids - self.act_idx)
                    loss_loc = self.loss_function(logits_loc, loc_ids - self.obj_idx)
                    loss: torch.Tensor = loss_act + loss_obj + loss_loc
                    loss.backward()

                    task.optimizer.step()

                    loss_log = loss.item()
                    self.epoch_loss += loss_log * batch_size

                    wandb_log({'train/loss': loss_log})

        phase_acc = phase_corrects / len(loader.dataset)

        log_data = {
            f"{phase}/epoch": self.epoch_id + 1,
            f"{phase}/epoch_acc": phase_acc
        }

        if is_train:
            self.epoch_loss = self.epoch_loss / len(loader.dataset)
            task.scheduler.step()
            print("Epoch Loss: ", self.epoch_loss)
            log_data[f"{phase}/epoch_loss"] = self.epoch_loss

        print(f"{phase.capitalize()} Accuracy: ", phase_acc)

        wandb_log(log_data)

        return phase_acc

