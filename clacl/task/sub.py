from __future__ import annotations

from functools import cached_property
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generic, TypeVar
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pydantic import BaseModel, SerializeAsAny

from clacl.model.wavlm_cl import AdapterState, AdaptivePoolState
from clacl.model.wavlm_cl import add_task, ensure_task_head, cl_modules
from clacl.task.common import WavMLClassificationTask as TaskBase, TaskConfig
from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import WandbConfig
from clacl.task.phase import Phase, TrainPhase, ValidPhase, CurrentTaskTestPhase, OtherTaskTestPhase
from clacl.task.phase import PPhases, Phases, Info, BatchData
from clacl.model.wavlm_cl import AdaWavLMForSequenceClassification, CustomWavLMConfig
from clacl.util import get_device, logger, wandb_log_table

if TYPE_CHECKING:
    from transformers import BatchFeature

class DatasetConfig(BaseModel):
    type: str = "Dataset"
    path: Path = Path("data/dataset/")

TDatasets = dict[str, SerializeAsAny[DatasetConfig]]

class TaskDataConfig(BaseModel):
    dataset: str = "dataset"
    csv_path: Path | None = Path("data/task/csv")

class ModelConfig(BaseModel):
    e_adapter: AdapterState = AdapterState.CL
    l_adapter: AdapterState = AdapterState.CL
    head: AdapterState = AdapterState.CL

    layer_weights_only: bool = False
    # use layer weights without l adapter (follow l_adapter state)

    head_adaptive_pool: AdaptivePoolState = AdaptivePoolState.Missing
    head_adaptive_pool_size: int = 0

class TrainConfig(BaseModel):
    log_layer_weights: bool = True

class SubTaskConfig(TaskConfig):
    type: str = "SubTask"
    data: TaskDataConfig = TaskDataConfig()
    model: ModelConfig | None = None

    train: TrainConfig | None = None
    
    dataset: TDatasets | None = None

    seed: int | None = None  # shadow

    # epochs: int = 10
    # batch_size: int = 32

    # learning_rate: LearningRate | None = None
    # scheduler: Scheduler | None = None

    @classmethod
    def with_all_fields(cls):
        return cls(
            model=ModelConfig(
                e_adapter=AdapterState.TuneAll, 
                l_adapter=AdapterState.TuneAll, 
                head=AdapterState.TuneAll
            ), 
            train=TrainConfig(),
            seed=42
        )

# TDatasetConfig = TypeVar("TDatasetConfig", bound=DatasetConfig)
TSubTaskConfig = TypeVar("TSubTaskConfig", bound=SubTaskConfig)
TSubTaskInConfig = TypeVar("TSubTaskInConfig", bound=SubTaskConfig)

def _init_sub_config(sub_config: TSubTaskConfig, wandb_config: WandbConfig | None = None) -> SubConfig[TSubTaskConfig]:
    if TYPE_CHECKING:
        global SubConfig

    # task_name = sub_config.name
    task_config_type = type(sub_config)

    class SubConfig(BaseModel, Generic[TSubTaskInConfig]):
        if TYPE_CHECKING:
            task: TSubTaskInConfig = sub_config
        else:
            task: task_config_type = sub_config
        wandb: WandbConfig | None = wandb_config

    return SubConfig[task_config_type]()

class SubTask(TaskBase, Generic[TSubTaskConfig]):

    if TYPE_CHECKING:
        from clacl.task.common import Config
        
        model: AdaWavLMForSequenceClassification
        
        config: TSubTaskConfig

    task_id: int = -1
    # _label_start = 0  # accumulate for each task

    # @staticmethod
    # def task_id2name(task_id: int):
    #     return f"task{task_id:02d}"
    
    @classmethod
    def from_config(cls, sub: TSubTaskConfig, wandb: WandbConfig | None = None):
        task = cls()
        task._raw_config = _init_sub_config(sub, wandb)
        # task.config.dataset = dataset
        return task

    # @cached_property
    # def _raw_config(self):
    #     return _init_config(SubTaskConfig())
    @property
    def is_CL(self):
        return self.task_id != -1

    @property
    def name(self):
        return self.config.name
    
    @property
    def name_with_id(self):
        if not self.is_CL:
            return self.name
        return f"{self.name}{self.task_id:02d}"

    @property
    def dataset(self):
        assert self.config.dataset
        return self.config.dataset
    
    @cached_property
    def model_config(self):
        assert self.config.model
        return {
            "id2label": {},
            "label2id": {},
            "num_labels": 0,
            "classifier_proj_size": 256,
            "e_adapter_state": self.config.model.e_adapter,
            "l_adapter_state": self.config.model.l_adapter,
            "head_state": self.config.model.head,
            "layer_weights_only": self.config.model.layer_weights_only,
            "head_adaptive_pool": self.config.model.head_adaptive_pool,
            "head_adaptive_pool_size": self.config.model.head_adaptive_pool_size,
        }

    def _model(self):
        self.model = AdaWavLMForSequenceClassification.from_pretrained(self.config.pretrained_name, **self.model_config)
        return self.model

    def init_model(self):
        self._model()
        if not self.is_CL:
            logger.info(f"Directly run {self.name_with_id} without a task_id")
            self.add_module()

    def _trainer(self):
        return SubTrainer(self)
    
    def inherit(self, pre_task: SubTask):
        self.model = pre_task.model
        self.model.config = DynamicModelConfig(self.name_with_id, self.model.config, self.model_config)
        logger.debug(f"{self.name_with_id} inherit model from {pre_task.name_with_id}, override configs")

    # def add_module(self, task_id: int):
    def add_module(self, pre_task: SubTask | None = None):
        # self.model.add_task(self.task_id2name(task_id))
        if not pre_task:
            pre_task = self
        # self.model.add_task(pre_task.name_with_id)
        add_task(self.model, pre_task.name_with_id)
        self._optimizer()
        self._scheduler()

    @contextmanager
    def ensure_task_head(self, device: torch.device | None = None):
        # old_task = self.model.ensure_task_head(self.name_with_id, device)
        old_task = ensure_task_head(self.model, self.name_with_id, device)
        logger.debug(f"Ensure task head for {self.name_with_id}")
        yield
        if old_task:
            # self.model.ensure_task_head(old_task, device)
            ensure_task_head(self.model, old_task, device)
            logger.debug(f"Reset task head to {old_task}")


    def evaluate_for(self, target_task: SubTask):
        trainer = self._trainer()
        return trainer.evaluate_for(target_task)

TSubTask = TypeVar("TSubTask", bound=SubTask[TSubTaskConfig])

class SubTrainer(TrainerBase, Generic[TSubTask]):
    if TYPE_CHECKING:
        task: TSubTask

    @cached_property
    def loss_function(self):
        return nn.CrossEntropyLoss()
    
    @property
    def _phases(self) -> PPhases:
        return Phases(TrainPhase(self.one_batch), ValidPhase(self.one_batch), CurrentTaskTestPhase(self.one_batch))

    @cached_property
    def total_epochs(self) -> int:
        # return self.task.config.epochs
        raise NotImplementedError

    @cached_property
    def _layer_weights(self):
        return []

    def train(self):
        self.device = get_device()

        self.task.model.to(self.device)
        loaders = self.task._data_loaders

        # torch.backends.cudnn.benchmark = True

        phases = self._phases

        # self.total_epochs = self.task.config.epochs
        for self.epoch_id in tqdm(range(self.total_epochs), desc="Epoch", position=0):
            logger.info(f"Epoch: {self.epoch_id + 1} / {self.total_epochs}")
            self.one_phase(phases.train, loaders.train)
            self.one_phase(phases.valid, loaders.valid)
            self.log_layer_weights()
        self.one_phase(phases.test, loaders.test)

    def one_phase(self, phase: Phase, loader: DataLoader):
        task = self.task
        with phase.on_epoch(task, loader, self._epoch_info) as info:
            for inputs, targets in phase.tqdm(loader):
                with phase.on_batch(task):
                    info = phase.one_batch(self._batch_data(inputs, targets, phase, info))
                    # inputs = inputs.to(self.device)

                    # output: SequenceClassifierOutput = task.model(**inputs)
                    # logits = output.logits.cpu()

                    # loss = phase.loss(self.loss_function, logits, targets)

                    # _, predict = torch.max(logits.data, 1)
                    # phase.update(info,
                    #     total=targets.size(0),
                    #     correct=(predict == targets).sum().item(),
                    #     loss=loss.item()
                    # )
        phase.summary(info, self.epoch_id)
        return info

    def _epoch_info(self, phase: Phase, loader: DataLoader) -> Info | None:
        return None  # Can return an different Info object

    def _batch_data(self, inputs: BatchFeature, targets: torch.Tensor, phase: Phase, info: Info) -> BatchData:
        return {
            "trainer": self,
            "inputs": inputs,
            "targets": targets,
            "phase": phase,
            "info": info
        }

    def one_batch(self, data: BatchData) -> Info:
        raise NotImplementedError

    @property
    def _evaluate_for_phase(self) -> Phase:
        return OtherTaskTestPhase(self.task.name_with_id, self.one_batch)

    def evaluate_for(self, target_task: SubTask[TSubTaskConfig]) -> Info:
        # evaluate target_task (current trained) with self.task
        self_task = self.task
        self_task.inherit(target_task)

        if self_task.task_id <= target_task.task_id:
            # self_task.model.set_task(self_task.name_with_id)
            cl_modules.set_task(self_task.name_with_id)
            # test task id > trained task id => use adapter of trained task, except head
        
        self.device = get_device()
        with self_task.ensure_task_head(self.device):
            self.epoch_id = 0
            phase = self._evaluate_for_phase
            loader = self_task._data_loaders.test

            test_info = self.one_phase(phase, loader)

        return test_info
    
        # task_config = target_task.config.task
        # self_task = self.task
        # self_task.inherit(target_task)

        # if task_config.cil_state == CILState.No:
        #     if self_task.task_id <= target_task.task_id:
        #         self_task.model.set_task(self_task.task_id2name(self_task.task_id))  
        #         # task_id > cl.task_id => use adapter of current task
        # elif task_config.cil_state == CILState.AverageAdapter:
        #     logger.debug("]] before average:", self_task.model.classifier.current_task)
        #     self_task.model.set_average_task()
        #     logger.debug("]] after average:", self_task.model.classifier.current_task)


        # self.epoch_id = 0
        # self.device = get_device()
        # phase = OtherTaskTestPhase(self_task.task_id, self.one_batch)
        # loader = self_task._data_loaders.test

        # # if task_config.cil_state == CILState.MaximizeLogit:
        # #     test_info = self.one_phase_max_logit(phase, loader)
        # # else:
        # #     test_info = self.one_phase(phase, loader)
        # test_info = self.one_phase(phase, loader)

        # return test_info

    def log_layer_weights(self):
        task = self.task
        if not task.config.train:
            return
        if not task.config.train.log_layer_weights:
            return
        if not hasattr(task.model, "current_layer_weights"):
            return
        current_layer_weights = task.model.current_layer_weights
        logger.info(f"layer weights:  {current_layer_weights!r}")
        self._layer_weights.append(current_layer_weights)
        columns = [f"layer {i}" for i in range(current_layer_weights.size(0))]
        wandb_log_table(f"{task.name_with_id}_layer_weights", columns=columns, data=self._layer_weights)


# class Trainer(TrainerBase):

#     if TYPE_CHECKING:
#         task: KSSubTask

#     loss_function = ICTrainer.loss_function

#     train = KS_CLSubTrainer.train

#     one_phase = KS_CLSubTrainer.one_phase

#     def evaluate_for(self, target_task: KSSubTask):
#         task_config = target_task.config.task
#         self_task = self.task
#         self_task.inherit(target_task)

#         if task_config.cil_state == CILState.No:
#             if self_task.task_id <= target_task.task_id:
#                 self_task.model.set_task(task_id2name(self_task.task_id))  
#                 # task_id > cl.task_id => use adapter of current task
#         elif task_config.cil_state == CILState.AverageAdapter:
#             logger.debug("]] before average:", self_task.model.classifier.current_task)
#             self_task.model.set_average_task()
#             logger.debug("]] after average:", self_task.model.classifier.current_task)


#         self.epoch_id = 0
#         self.device = get_device()
#         phase = OtherTaskTestPhase(self_task.task_id)
#         loader = self_task._data_loaders.test

#         if task_config.cil_state == CILState.MaximizeLogit:
#             test_info = self.one_phase_max_logit(phase, loader)
#         else:
#             test_info = self.one_phase(phase, loader)

#         return test_info
    
#     def one_phase_max_logit(self, phase: Phase, loader: DataLoader):
#         task = self.task
#         with phase.on_epoch(task, loader) as info:
#             for inputs, targets in phase.tqdm(loader):
#                 with phase.on_batch(task):
#                     inputs = inputs.to(self.device)

#                     max_logits_task = ""
#                     max_logits = None
#                     max_logits_value = 0

#                     for task_name in task.model._task_names_gen():
#                         logger.debug(f"Under {task_name} model")
#                         task.model.set_task(task_name)

#                         output: SequenceClassifierOutput = task.model(**inputs)
#                         logits = output.logits.cpu()

#                         if (logits_max := logits.max().item()) > max_logits_value:
#                             max_logits_task = task_name
#                             max_logits = logits
#                             max_logits_value = logits_max
#                             logger.debug(f"{max_logits_task = }, {max_logits_value = }")

#                     logger.debug(f"Final {max_logits_task = }, {max_logits_value = }")

#                     # loss = phase.loss(self.loss_function, logits, targets)
#                     loss = phase.loss(self.loss_function, max_logits, targets)

#                     # _, predict = torch.max(logits.data, 1)
#                     _, predict = torch.max(max_logits.data, 1)
#                     phase.update(info,
#                         total=targets.size(0),
#                         correct=(predict == targets).sum().item(),
#                         loss=loss.item()
#                     )
#         phase.summary(info, self.epoch_id)
#         return info

def edit_model(model: AdaWavLMForSequenceClassification):
    down_param = []
    # encoder_param = []
    layernorm_param = []
    # layerweight_param = []
    adapter_param = []
    adapter_to_output_param = []
    adapter_to_output_layer_weights_param=[]
    
    frozen, tuned = 0, 0
    
    layer_names = [f'layers.{k}' for k in range(0,12)]
    for name, param in model.named_parameters():
        flag = any(idx in name for idx in layer_names)
        is_not_frozen = param.requires_grad
        
        if is_not_frozen and ('projector' in name or 'classifier' in name):
            logger.info(f'down: {name}')
            tuned += param.numel()
            down_param.append(param)
        elif is_not_frozen and 'adapter_to_output_layer_weights' in name:
            logger.info(f'adapter_to_output_layer_weights: {name}')
            tuned += param.numel()
            adapter_to_output_layer_weights_param.append(param)
        # elif ('encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder):
        elif is_not_frozen and ('encoder.layers' in name and 'layer_norm' in name and flag):
            logger.info(f'layer_norm: {name}')
            tuned += param.numel()
            layernorm_param.append(param)
        elif is_not_frozen and 'adapter_to_output' in name:
            logger.info(f'adapter_output: {name}')
            tuned += param.numel()
            adapter_to_output_param.append(param)
        elif is_not_frozen and 'adapter_layer' in name:
            logger.info(f'adapter_layer: {name}')
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
            logger.info(f'frozen: {name}')
            frozen += param.numel()
            param.requires_grad = False

    total = tuned + frozen
    logger.info(f"num of tuned params: {tuned} / {total} ({tuned / total * 100:.2f}%)")
    return down_param, layernorm_param, adapter_param, adapter_to_output_layer_weights_param, adapter_to_output_param

class DynamicModelConfig:

    def __init__(self, task_name: str, config: CustomWavLMConfig, over_map: dict | None = None):
        self._task_name = task_name
        if isinstance(config, DynamicModelConfig):
            self._config = config._config
        else:
            self._config = config
        self._over_map = over_map or {}

    def __getattr__(self, name: str):
        if name in self._over_map:
            return self._over_map[name]
        return getattr(self._config, name)
    
    def __setattr__(self, name: str, value):
        if name in ['_config', '_over_map', '_task_name']:
            super().__setattr__(name, value)
        else:
            setattr(self._config, name, value)
    
    def __delattr__(self, name: str):
        if name in ['_config', '_over_map', '_task_name']:
            super().__delattr__(name)
        else:
            delattr(self._config, name)

    def __repr__(self):
        return f"DynamicModelConfig({self._task_name!r}, {self._config!r}, {self._over_map!r})"

