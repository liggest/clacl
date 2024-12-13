from typing import TYPE_CHECKING
from functools import cached_property

from pydantic import ConfigDict, Field
import torch.nn as nn
from transformers.models.wavlm import WavLMForSequenceClassification

from clacl.model.wavml_cl import AdapterState, AdaptivePoolState
from clacl.model.wavml_cl import ModelAdaptiveAvgPool1d, ModelAdaptiveMaxPool1d
# from clacl.task.common import WavMLClassificationTask as TaskBase
# from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import TaskConfig, _init_config
from clacl.task.sub import ModelConfig, TrainConfig, TSubTaskConfig
from clacl.task.CLSCL import UDatasetConfig, USubTaskConfig
from clacl.task.CLSCL import Datasets, SubConfigs
from clacl.task.CLSCL import CLTask as TaskBase, CLTrainer as TrainerBase
from clacl.task.finetune import LearningRate
from clacl.task.finetune import FinetuningSubTaskWrapper
from clacl.util import logger

# SubConfigs = {k: v.__class__.with_all_fields() for k, v in SubConfigs.items()}

class FinetuningModelConfig(ModelConfig):
    e_adapter: AdapterState = AdapterState.Missing
    l_adapter: AdapterState = AdapterState.Missing
    head: AdapterState = AdapterState.TuneAll

    layer_weights_only: bool = Field(default=False, alias="use_weighted_layer_sum")

    head_adaptive_pool: AdaptivePoolState = AdaptivePoolState.Avg
    head_adaptive_pool_size: int = 128

class FinetuningCLConfig(TaskConfig):
    model_config = ConfigDict(extra = "allow")

    name: str = "FinetuningCL"

    dataset: dict[str, UDatasetConfig] = Datasets

    model: ModelConfig = FinetuningModelConfig()
    train: TrainConfig = TrainConfig()
    learning_rate: LearningRate = LearningRate()

    sub: dict[str, USubTaskConfig] = SubConfigs
    sequence: list[str] = ["KS_sub", "IC_sub", "ER_sub", "AcC_sub", "LID_sub"]

    pretrained_name: str | None = None  # shadow
    optimizer: str | None = None

class FinetuningCLTask(TaskBase):
    if TYPE_CHECKING:
        from clacl.task.common import Config

        _raw_config: Config[FinetuningCLConfig]
        config: FinetuningCLConfig

    @cached_property
    def _raw_config(self):
        return _init_config(FinetuningCLConfig())
    
    def init_subs(self):
        super().init_subs()
        self.subs = { name: FinetuningCLSubTaskWrapper(task, self) for name, task in self.subs.items() }
        logger.debug(f"All sub tasks wrppered in {FinetuningCLSubTaskWrapper.__name__}")

    def _trainer(self):
        return FinetuningCLTrainer(self)
    
class FinetuningCLTrainer(TrainerBase):
    if TYPE_CHECKING:
        task: FinetuningCLTask

def _classifier(model: WavLMForSequenceClassification, model_config: ModelConfig):
    assert model_config.head_adaptive_pool != AdaptivePoolState.Missing
    if model_config.head_adaptive_pool == AdaptivePoolState.Max:
        pool = ModelAdaptiveMaxPool1d(model)
    else:
        pool = ModelAdaptiveAvgPool1d(model)
    return nn.Sequential(
        nn.Linear(model.config.classifier_proj_size, model_config.head_adaptive_pool_size),
        pool,
        # nn.Flatten(),
    )

class FinetuningCLSubTaskWrapper(FinetuningSubTaskWrapper[TSubTaskConfig]):

    def _model(self):
        super()._model()
        self.model.classifier = _classifier(self.model, self._task.config.model)
        return self.model
