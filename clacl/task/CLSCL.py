"""
Continual Learning for Speech CLassification
"""

from typing import TYPE_CHECKING, Annotated
from functools import cached_property
from itertools import pairwise

import torch
from pydantic import ConfigDict, Field, BaseModel

from clacl.model.wavlm_cl import AdapterState, CLInitSpecials, cl_modules
from clacl.task.common import WavMLClassificationTask as TaskBase
from clacl.task.common import WavMLClassificationTrainer as TrainerBase
from clacl.task.common import TaskConfig, _init_config
from clacl.task.sub import SubTask, TSubTaskConfig
from clacl.task.sub import ModelConfig, TrainConfig
from clacl.util import wandb_log, logger

from clacl.task.KS_sub import KSSubTask, KSSubConfig, SpeechCommandsConfig
from clacl.task.IC_sub import ICSubTask, ICSubConfig, FluentSpeechCommandsConfig
from clacl.task.ER_sub import ERSubTask, ERSubConfig, IEMOCAPConfig
from clacl.task.AcC_sub import AcCSubTask, AcCSubConfig, AccentDBConfig
from clacl.task.LID_sub import LIDSubTask, LIDSubConfig, VoxForgeConfig
from clacl.task.FSD_sub import FSDSubTask, FSDSubConfig, ASVspoof2019Config

UDatasetConfig = Annotated[
    SpeechCommandsConfig | FluentSpeechCommandsConfig | IEMOCAPConfig | AccentDBConfig | VoxForgeConfig | ASVspoof2019Config, # | DatasetConfig, 
    Field(discriminator="type")  # TODO: add more datasets
]
USubTaskConfig = Annotated[
    KSSubConfig | ICSubConfig | ERSubConfig | AcCSubConfig | LIDSubConfig | FSDSubConfig, # | SubTaskConfig,
    Field(discriminator="type")  # TODO: add more sub tasks
]

Datasets: dict[str, UDatasetConfig] = {
    "SpeechCommands": SpeechCommandsConfig(),
    "FluentSpeechCommands": FluentSpeechCommandsConfig(),
    "IEMOCAP": IEMOCAPConfig(),
    "AccentDB": AccentDBConfig(),
    "VoxForge": VoxForgeConfig(),
    "ASVspoof2019": ASVspoof2019Config(),
}

SubConfigs: dict[str, USubTaskConfig] = {
    "KS_sub": KSSubConfig(),
    "IC_sub": ICSubConfig(),
    "ER_sub": ERSubConfig(),
    "AcC_sub": AcCSubConfig(),
    "LID_sub": LIDSubConfig(),
    "FSD_sub": FSDSubConfig(),
}

SubClasses: dict[str, type[SubTask[TSubTaskConfig]]] = {
    "KSSubTask": KSSubTask,
    "ICSubTask": ICSubTask,
    "ERSubTask": ERSubTask,
    "AcCSubTask": AcCSubTask,
    "LIDSubTask": LIDSubTask,
    "FSDSubTask": FSDSubTask,
}

class CLConfig(BaseModel):
    e_adapter_init: list[str | CLInitSpecials] = [CLInitSpecials.Self]
    l_adapter_init: list[str | CLInitSpecials] = [CLInitSpecials.Self]
    layer_weights_init: list[str | CLInitSpecials] = [CLInitSpecials.Self]
    head_init: list[str | CLInitSpecials] = [CLInitSpecials.Self]

    head_expanding: bool = False

class CLSCLConfig(TaskConfig):
    model_config = ConfigDict(extra = "allow")

    name: str = "CLSCL"

    dataset: dict[str, UDatasetConfig] = Datasets
    
    model: ModelConfig = ModelConfig(e_adapter=AdapterState.CL, l_adapter=AdapterState.CL, head=AdapterState.CL)
    cl: CLConfig = CLConfig()
    train: TrainConfig = TrainConfig()

    sub: dict[str, USubTaskConfig] = SubConfigs
    sequence: list[str] = ["KS_sub", "IC_sub", "ER_sub", "AcC_sub", "LID_sub"]

    pretrained_name: str | None = None  # shadow
    optimizer: str | None = None

def _fill_sub_config(sub_config: USubTaskConfig, cl_config: CLSCLConfig):
    sub_config.dataset = cl_config.dataset
    sub_config.seed = cl_config.seed
    sub_config.model = cl_config.model
    sub_config.train = cl_config.train
    return sub_config

class CLTask(TaskBase):
    if TYPE_CHECKING:
        from clacl.task.common import Config

        _raw_config: Config[CLSCLConfig]
        config: CLSCLConfig

    @cached_property
    def _raw_config(self):
        return _init_config(CLSCLConfig())

    @property
    def current_task(self):
        return self.tasks[self.task_id]

    @property
    def model(self):
        return self.current_task.model

    @property
    def optimizer(self):
        return self.current_task.optimizer
    
    @property
    def scheduler(self):
        return self.current_task.scheduler

    @property
    def _model_config_extra(self):
        config = self.config
        return {
            "cl_e_adapter_init": config.cl.e_adapter_init,
            "cl_l_adapter_init": config.cl.l_adapter_init,
            "cl_layer_weights_init": config.cl.layer_weights_init,
            "cl_head_init": config.cl.head_init,
            "head_expanding": config.cl.head_expanding,
        }

    def init_subs(self):
        cl_config = self.config
        sub_config = cl_config.sub
        # dataset = self.config.dataset
        wandb = self._raw_config.wandb
        self.subs = { c.name: SubClasses[c.type].from_config(_fill_sub_config(c, cl_config), wandb) for c in sub_config.values() }
        logger.debug(f"Available sub tasks: {self.subs!r}")

    def init_task_sequence(self):
        self.tasks = [self.subs[name] for name in self.config.sequence]
        self.task_id = 0
        self.current_task.task_id = 0  # ensure the first task is initialized with CL manner
        logger.info(f"Current task sequence: {' >> '.join(t.name for t in self.tasks)}")

    def init_model(self):
        self.init_subs()
        self.init_task_sequence()
        self.current_task.model_config.update(self._model_config_extra)
        self.current_task.init_model()

    def _trainer(self):
        return CLTrainer(self)

class CLTrainer(TrainerBase):
    
    if TYPE_CHECKING:
        task: CLTask

    def train(self):
        self.learned_averages = []
        
        cl = self.task
        last_task = None
        total_tasks = len(cl.tasks)

        self.accuracy_mat = torch.zeros((total_tasks, total_tasks), dtype=torch.float64)

        for cl.task_id, task in enumerate(cl.tasks):
            task.task_id = cl.task_id
            logger.info(f"Task: {task.name_with_id} ({cl.task_id + 1} / {total_tasks})")
            # logger.debug(repr(task.config.model_dump()))
            # print(, f"({len(task.classes)} classes)")
            
            if last_task:
                task.inherit(last_task)
            # task.add_module(cl.task_id)
            task.add_module()
            last_cl_module = None
            if last_cl_module := cl_modules.get("head", {}).get("classifier"):
                logger.info(f">> before train: {last_cl_module.current_task}")
            
            task.train()

            # task.model.set_task_grad(freeze=True, f=lambda adapter: adapter.state != AdapterState.TuneAll)
            cl_modules.set_task_grad(freeze=True, f=lambda adapter, *_: adapter.state != AdapterState.TuneAll)

            self.evaluate_all_for(task)

            # task.model.set_task(task.task_id2name(task.task_id))
            # task.model.set_task(task.name_with_id)
            cl_modules.set_task(task.name_with_id)
            if last_cl_module:
                logger.info(f">> after train: {last_cl_module.current_task}")
            last_task = task
        
    def evaluate_all_for(self, current_task: SubTask[TSubTaskConfig]):
        cl = self.task
        # learned_average_accuracy = 0
        cid = current_task.task_id
        for task_id, test_task in enumerate(cl.tasks):
            test_task.task_id = task_id
            test_info = test_task.evaluate_for(current_task)

            self.accuracy_mat[cid, task_id] = test_info.accuracy


            # if test_task.task_id <= current_task.task_id:
            #     learned_average_accuracy += test_accuracy
        
        # learned_average_accuracy /= (cl.task_id + 1)
        learned_average_accuracy = Metrics.learned_average(self.accuracy_mat, cid)

        self.learned_averages.append(learned_average_accuracy)
        
        backward_transfer = Metrics.backward_transfer(self.accuracy_mat, cid)

        wandb_log({
            "test/learned_average_acc": learned_average_accuracy,
            "test/backward_transfer": backward_transfer,
            "test/backward_transfer_gem": Metrics.backward_transfer_gem(self.accuracy_mat, cid),
            "test/backward_transfer_kws": Metrics.backward_transfer_kws(self.learned_averages),
            "test/forward_transfer_gem": Metrics.forward_transfer_gem(self.accuracy_mat, cid),
            "test/forward_transfer": Metrics.forward_transfer(self.accuracy_mat, cid),
            "test/average_forgetting": Metrics.forgetting(self.accuracy_mat, cid),
        })


class Metrics:

    @staticmethod
    def learned_average(accuracy_mat: torch.Tensor, cid: int):
        return accuracy_mat[cid, :cid + 1].mean().item()

    @staticmethod
    def backward_transfer_kws(learned_averages: list[float]):
        if len(learned_averages) <= 1:
            return 0
        return sum(a1 - a0 for a0, a1 in pairwise(learned_averages)) / (len(learned_averages) - 1)

    @staticmethod
    def backward_transfer_gem(accuracy_mat: torch.Tensor, cid: int):
        if cid < 1:
            return 0
        # previous tasks, exclude current task
        # accuracy_mat.diag() are accuracy of tasks when they were just trained 
        return (accuracy_mat[cid, :cid] - accuracy_mat.diag()[:cid]).mean().item()
        
    
    @staticmethod
    def backward_transfer(accuracy_mat: torch.Tensor, cid: int):
        if cid < 1:
            return 0
        task_count = cid + 1
        sub = accuracy_mat[:task_count, :task_count]
        lower_triangle = (sub - sub.diag()).tril(diagonal=-1)  # without diagonal
        # return lower_triangle[lower_triangle!=0].mean().item()  # may be incorrect if some values in thelower triangle are zero
        return (lower_triangle.sum() / (task_count * (task_count - 1) / 2)).item()
        

    @staticmethod
    def forward_transfer_gem(accuracy_mat: torch.Tensor, cid: int, init_accuracy: torch.Tensor | None = None):
        task_count = cid + 1
        diag = accuracy_mat[:task_count].diag(diagonal=1)  # accuracy_mat[i-1, i]
        if init_accuracy is None:
            init_accuracy = torch.zeros_like(diag, dtype=torch.float64)
        else:
            init_accuracy = init_accuracy[:diag.size(0)]
        return (diag - init_accuracy).mean().item()

    @staticmethod
    def forward_transfer(accuracy_mat: torch.Tensor, cid: int):
        task_count = cid + 1
        upper_triangle = accuracy_mat[:task_count].triu(diagonal=1)  # without diagonal
        # return upper_triangle[upper_triangle!=0].mean().item()
        return (upper_triangle.sum() / (task_count * (task_count - 1) / 2)).item()

    @staticmethod
    def forgetting(accuracy_mat: torch.Tensor, cid: int):
        if cid < 1:
            return 0
        max_previous, _ = accuracy_mat[:cid, :cid].max(dim=0)
        return(max_previous - accuracy_mat[cid, :cid]).mean().item()
