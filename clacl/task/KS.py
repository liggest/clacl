# Code partially derived from https://github.com/s3prl/s3prl/blob/main/s3prl/problem/common/superb_ks.py

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from pathlib import Path
from functools import cached_property

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ExponentialLR
from pydantic import BaseModel
# from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

from clacl.task.common import WavMLClassificationTask as TaskBase
from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import TaskConfig, _init_config
from clacl.data.common import DataPieceCollator as Collator, DataLoaders
from clacl.data.speech_commands import Dataset, DatasetWithSilence
from clacl.task.IC import LearningRate, Task as ICTask, Trainer as ICTrainer
from clacl.util import get_device, wandb_log

if TYPE_CHECKING:
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.modeling_outputs import SequenceClassifierOutput
    from clacl.task.common import Config

class Scheduler(BaseModel):
    type: str = "ExponentialLR"
    gamma: float = 0.9

class KSConfig(TaskConfig):
    name: str = "KS"

    csv_path: Path = Path("data/KS/csv")
    data_path: Path = Path("data/KS/speech_commands_v0.01")
    test_path: Path = Path("data/KS/speech_commands_test_set_v0.01")

    # pretrained_name: str = "microsoft/wavlm-base-plus"
    # epochs: int = 7
    total_steps: int = 200000
    eval_steps: int = 5000
    # gradient_clipping: float = 1.0
    batch_size: int = 32
    # optimizer: str = "Adam"
    learning_rate: LearningRate = LearningRate()
    scheduler: Scheduler = Scheduler()

# def _init_config():
#     class Config(file_config_base("data/KS/config.toml")):
#         KS: KSConfig = KSConfig()
#         wandb: WandbConfig = WandbConfig()
    
#     return Config()

class Task(TaskBase):

    if TYPE_CHECKING:
        scheduler: ExponentialLR

        _raw_config: Config[KSConfig]
        config: KSConfig

    @cached_property
    def _raw_config(self):
        return _init_config(KSConfig())

    # @cached_property
    # def pre_config(self):
    #     return Wav2Vec2Config.from_pretrained("superb/wav2vec2-large-superb-ks")

    @cached_property
    def extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-ks")
    
    @property
    def _data_loaders(self):
        data_path = self.config.data_path
        test_path = self.config.test_path
        csv_path = self.config.csv_path
        train_dataset = DatasetWithSilence(data_path, csv_path / "train.csv")
        val_dataset = DatasetWithSilence(data_path, csv_path / "valid.csv")
        test_dataset = Dataset(test_path, csv_path / "test.csv")

        collator = Collator(self.extractor)

        batch_size = self.config.batch_size

        train_weights = train_dataset.sample_weights
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, sampler=train_sampler, num_workers=12)

        val_weights = val_dataset.sample_weights
        val_sampler = WeightedRandomSampler(val_weights, len(val_weights))
        val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, sampler=val_sampler, num_workers=12)

        test_loader = DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=12)

        # return {'train':train_loader, 'val':val_loader}
        return DataLoaders(train_loader, val_loader, test_loader)


    @property
    def model_config(self):
        from clacl.data.speech_commands import SUPERB_CLASSES
        # ks_sub = CLASSES[:12]  # superb 12 class
        ks_sub = SUPERB_CLASSES
        label2id = {label: Dataset.label2id[label] for label in ks_sub}
        return {
            "id2label": {str(i): label for label, i in label2id.items()},
            "label2id": label2id,
            "num_labels": len(ks_sub),
            "classifier_proj_size": 256
        }

    _model = ICTask._model
    _edit_model = ICTask._edit_model
    _optimizer = ICTask._optimizer

    def _scheduler(self):
        self.scheduler = ExponentialLR(self.optimizer, self.config.scheduler.gamma)
        return self.scheduler

    def _trainer(self):
        return Trainer(self)

class Trainer(TrainerBase):

    if TYPE_CHECKING:
        task: Task
    
    loss_function = ICTrainer.loss_function
        
    def train(self):
        self.device = get_device()

        self.task.model.to(self.device)
        loaders = self.task._data_loaders

        torch.backends.cudnn.benchmark = True

        # self.total_epochs = self.task.config.epochs
        self.total_steps = self.task.config.total_steps
        eval_steps = self.task.config.eval_steps
        self.epoch_id = 0

        for step_id in tqdm(
            self.train_steps_gen(loaders.train), 
            desc="Steps", total=self.total_steps, position=0
        ):
            current_steps = step_id + 1

            if current_steps % eval_steps == 0:
                self.evaluate("valid", loaders.valid)
                self.task.model.train()  # back to train mode
            if current_steps >= self.total_steps:
                print(f"{current_steps} / {self.total_steps} Finished")
                break  # end

        self.evaluate("test", loaders.test)
        # for epoch_id in tqdm(range(self.total_epochs), desc="Epoch", position=0):
        #     print(f"Epoch: {epoch_id} / {self.total_epochs}")
        #     self.epoch_id = epoch_id
        #     self.train_phase(loaders["train"])
        #     self.eval_phase(loaders["val"])

        # test_loaders = self.task._data_loaders
        # self.test_phase(test_loaders["val"])
        #     for phase in ("train", "val"):
        #         self.one_phase(phase, loaders[phase])
        # self.one_phase("test", loaders["test"])

    def train_steps_gen(self, loader: DataLoader):
        task = self.task
        task.model.train()
        step_id = 0
        while True:
            train_loss = 0
            train_total = 0
            train_correct = 0
            train_accuracy = 0

            try:
                for step_id, (inputs, targets) in enumerate(tqdm(
                    loader, desc="Batch [train]", position=1, leave=False
                    ), start=step_id):
                    inputs: BatchFeature
                    targets: torch.LongTensor
                    inputs = inputs.to(self.device)

                    task.optimizer.zero_grad()

                    output: SequenceClassifierOutput = task.model(**inputs)
                    logits = output.logits.cpu()
                    loss: torch.Tensor = self.loss_function(logits, targets)
                    loss.backward()
                    
                    # grad_norm = nn.utils.clip_grad_norm_(task.model.parameters(), task.config.gradient_clipping)

                    # if torch.isnan(grad_norm).any():
                    #     print(f"> grad_norm has NaN at step {step_id + 1} <")

                    task.optimizer.step()

                    # task.scheduler.step()

                    train_loss += loss.item() / len(loader)
                    _, predict = torch.max(logits.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predict == targets).sum().item()
                    train_accuracy = train_correct / train_total
                    
                    wandb_log({'train/loss': loss.item()})

                    yield step_id # one step end
            finally:
                print("Epoch Train Loss: ", train_loss)
                print("Epoch Train Acc:  ", train_accuracy)
                wandb_log({
                    "train/epoch": self.epoch_id + 1,
                    "train/epoch_acc": train_accuracy,
                    "train/epoch_loss": train_loss
                })

                task.scheduler.step()

                self.epoch_id += 1

    def evaluate(self, phase: Literal["valid", "test"], loader: DataLoader):
        task = self.task
        task.model.eval()

        if phase == "valid":
            eval_loss = 0
        eval_total = 0
        eval_correct = 0
        eval_accuracy = 0

        for inputs, targets in tqdm(loader, desc=f"Batch [{phase}]", position=1, leave=False):
            inputs: BatchFeature
            targets: torch.LongTensor
            with torch.no_grad():
                inputs = inputs.to(self.device)
                # targets = targets.to(self.device)

                output: SequenceClassifierOutput = task.model(**inputs)
                logits = output.logits.cpu()
                if phase == "valid":
                    loss: torch.Tensor = self.loss_function(logits, targets)

                    eval_loss += loss.item() / len(loader)
                _, predict = torch.max(logits.data, 1)
                eval_total += targets.size(0)
                eval_correct += (predict == targets).sum().item()
                eval_accuracy = eval_correct / eval_total
        
        log_data = {
            f"{phase}/epoch": self.epoch_id + 1,
            f"{phase}/epoch_acc": eval_accuracy,
        }

        Phase = phase.capitalize()
        if phase == "valid":
            print(f"{Phase} Loss: ", eval_loss)
            log_data[f"{phase}/epoch_loss"] = eval_loss
        print(f"{Phase} Acc:  ", eval_accuracy)

        wandb_log(log_data)
