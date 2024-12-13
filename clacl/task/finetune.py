"""
    Plain Finetuning
"""

from typing import TYPE_CHECKING
from pathlib import Path
from functools import cached_property

from torch.optim import Adam
from pydantic import BaseModel, TypeAdapter
import tomli
import tomli_w
from transformers.models.wavlm import WavLMForSequenceClassification

from clacl.model.wavlm_cl import AdapterState
from clacl.task.common import WavMLClassificationTask as TaskBase
from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import TaskConfig, _init_config
from clacl.task.sub import SubTask, SubTaskConfig
from clacl.task.sub import TSubTaskConfig
from clacl.task.CLSCL import USubTaskConfig, SubClasses
from clacl.config import cli_config
from clacl.util import logger

class SubConfigFile(BaseModel):
    config: Path = Path("data/IC_sub/config_test.toml")

class LearningRate(BaseModel):
    down: float | None = 5e-4
    encoder: float = 1e-4

class FinetuningConfig(TaskConfig):
    name: str = "Finetuning"

    sub: SubConfigFile | USubTaskConfig = SubConfigFile()

    learning_rate: LearningRate = LearningRate()

    pretrained_name: str | None = None  # shadow
    optimizer: str | None = None

class FinetuningTask(TaskBase):
    if TYPE_CHECKING:
        from clacl.task.common import Config

        _raw_config: Config[FinetuningConfig]
        config: FinetuningConfig

    @cached_property
    def _raw_config(self):
        return _init_config(FinetuningConfig())
    
    @cached_property
    def config(self):
        _config = self._raw_config.task
        self._init_sub_config(_config)
        return _config

    @property
    def current_task(self):
        return self.sub_task

    @property
    def model(self):
        return self.current_task.model

    @property
    def optimizer(self):
        return self.current_task.optimizer
    
    @property
    def scheduler(self):
        return self.current_task.scheduler
    
    def _init_sub_config(self, config: FinetuningConfig):
        if cli_config().dump:
            # dump original config, no modification
            return
        if isinstance(config.sub, SubConfigFile):
            logger.info(f"Found sub task config in finetuning refer to another config file {config.sub.config.as_posix()}, loading it...")
            with config.sub.config.open("rb") as f:
                sub_config_dict = tomli.load(f)
            sub_config: USubTaskConfig = TypeAdapter(USubTaskConfig).validate_python(sub_config_dict.get("task"))
            config.sub = sub_config
            logger.info(f"Loaded sub task config with name: {sub_config.name!r} and type {sub_config.type!r}")
            self._init_sub_model_config(config.sub)
            self._init_sub_learning_rate_config(config, config.sub)
            self._init_sub_train_config(config.sub)

            dump_path: Path = next(self._raw_config._config_paths())
            config_dict = self._raw_config.model_dump(mode = "json", exclude_none = True)
            with dump_path.open("wb") as f:
                tomli_w.dump(config_dict, f)

            logger.info(f"Overriding finetuning config with sub task config, dumping it to {dump_path.as_posix()}")

    def _init_sub_model_config(self, sub_config: USubTaskConfig):
        sub_config.model.e_adapter = AdapterState.Missing
        sub_config.model.l_adapter = AdapterState.Missing
        sub_config.model.head = AdapterState.TuneAll
        logger.debug(f"Overriding model config of sub task to: {sub_config.model!r}")

    def _init_sub_learning_rate_config(self, config: FinetuningConfig, sub_config: USubTaskConfig):
        if hasattr(sub_config, "learning_rate"):
            sub_config.learning_rate.adapter_to_output = 0
            sub_config.learning_rate.adapter_layer_weights = 0
            sub_config.learning_rate.adapter_ff = 0
            sub_config.learning_rate.layer_norm = 0
            if config.learning_rate.down is None:
                config.learning_rate.down = sub_config.learning_rate.down
            else:
                sub_config.learning_rate.down = config.learning_rate.down
            logger.debug(f"Overriding learning rate config of sub task to: {sub_config.learning_rate!r}")

    def _init_sub_train_config(self, sub_config: USubTaskConfig):
        sub_config.train.log_layer_weights = False
        logger.debug(f"Overriding train config of sub task to: {sub_config.train!r}")

    def init_sub(self):
        sub_config = self.config.sub
        assert isinstance(sub_config, SubTaskConfig)
        sub_task = SubClasses[sub_config.type].from_config(sub_config, self._raw_config.wandb)
        logger.debug(f"Current sub task: {sub_config.type!r}")
        self.sub_task = FinetuningSubTaskWrapper(sub_task, self)

    def init_model(self):
        self.init_sub()
        self.current_task.init_model()

    def _trainer(self):
        return FinetuningTrainer(self)

class FinetuningTrainer(TrainerBase):
    if TYPE_CHECKING:
        task: FinetuningTask

    def train(self):
        sub_task = self.task.current_task
        logger.debug(f"Finetuning sub task: {sub_task.name_with_id!r}")
        
        sub_task.train()

class FinetuningSubTaskWrapper(SubTask[TSubTaskConfig]):
    
    if TYPE_CHECKING:
        model: WavLMForSequenceClassification

    def __init__(self, task: SubTask[TSubTaskConfig], finetuning_task: FinetuningTask):
        super().__init__()
        self._task = task
        self._ft = finetuning_task

    @property
    def _raw_config(self):
        return self._task._raw_config
    
    @property
    def config(self):
        return self._task.config

    @property
    def _data_loaders(self):
        return self._task._data_loaders

    @property
    def task_id(self):
        return self._task.task_id

    @task_id.setter
    def task_id(self, val: int):
        self._task.task_id = val
    
    @task_id.deleter
    def task_id(self):
        del self._task.task_id

    @property
    def model_config(self):
        _model_config = self._task.model_config
        # only these parameters can be passed to model
        return {
            "id2label": _model_config["id2label"],
            "label2id": _model_config["label2id"],
            "num_labels": _model_config["num_labels"],
            "classifier_proj_size": _model_config["classifier_proj_size"],
            "use_weighted_layer_sum": _model_config["layer_weights_only"]
        }

    def _model(self):
        self.model = WavLMForSequenceClassification.from_pretrained(self.config.pretrained_name, **self.model_config)
        return self.model

    def _trainer(self):
        return self._task._trainer()

    def add_module(self, pre_task: SubTask | None = None):
        # if not pre_task:
        #     pre_task = self
        # self.model.add_task(pre_task.name_with_id)
        self._optimizer()
        self._scheduler()

    def _optimizer(self):
        (down_param, 
        encoder_param
        ) = edit_model(self.model)
        learning_rate = self._ft.config.learning_rate
        self.optimizer = Adam([
            {'params': down_param, 'lr': learning_rate.down},
            {'params': encoder_param, 'lr': learning_rate.encoder},
        ])
        return self.optimizer

    def _scheduler(self):
        return self._task._scheduler()

    def __getattr__(self, name: str):
        # logger.debug(f"__getattr__: {name!r}")
        return getattr(self._task, name)
    
    def __setattr__(self, name: str, value):
        if name in ["_task", "_ft"]:
            super().__setattr__(name, value)
        else:
            setattr(self._task, name, value)

    def __delattr__(self, name: str):
        if name in ["_task", "_ft"]:
            super().__delattr__(name)
        else:
            delattr(self._task, name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._task!r}, {self._ft!r})"

def edit_model(model: WavLMForSequenceClassification):
    down_param = []
    encoder_param = []
    # layernorm_param = []
    # layerweight_param = []
    # adapter_param = []
    # adapter_to_output_param = []
    # adapter_to_output_layer_weights_param=[]
    
    frozen, tuned = 0, 0
    
    # layer_names = [f'layers.{k}' for k in range(0,12)]
    layer_names = [str(i) for i in range(0, 12)]
    for name, param in model.named_parameters():
        flag = any(idx in name for idx in layer_names)
        is_not_frozen = param.requires_grad
        
        if is_not_frozen and ('projector' in name or 'classifier' in name):
            logger.info(f'down: {name}')
            tuned += param.numel()
            down_param.append(param)
        # elif is_not_frozen and 'adapter_to_output_layer_weights' in name:
        #     logger.info(f'adapter_to_output_layer_weights: {name}')
        #     tuned += param.numel()
        #     adapter_to_output_layer_weights_param.append(param)
        # elif ('encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder):
        # elif is_not_frozen and ('encoder.layers' in name and 'layer_norm' in name and flag):
        #     logger.info(f'layer_norm: {name}')
        #     tuned += param.numel()
        #     layernorm_param.append(param)
        # elif is_not_frozen and 'adapter_to_output' in name:
        #     logger.info(f'adapter_output: {name}')
        #     tuned += param.numel()
        #     adapter_to_output_param.append(param)
        # elif is_not_frozen and 'adapter_layer' in name:
        #     logger.info(f'adapter_layer: {name}')
        #     tuned += param.numel()
        #     adapter_param.append(param)
        elif 'encoder.layers' in name and flag:
            logger.info(f'encoder: {name}')
            tuned += param.numel()
            encoder_param.append(param)
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
    # return down_param, layernorm_param, adapter_param, adapter_to_output_layer_weights_param, adapter_to_output_param
    return down_param, encoder_param
