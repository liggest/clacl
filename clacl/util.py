
import random
import logging
import numpy as np
import torch
import wandb

from typing_extensions import Self

from clacl.task.common import WandbConfig

def fix_seed(seed):
    logger.info(f"Fixing seed to {seed!r}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Singleton:

    _instance: Self = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def instance(self):
        return self._instance

def wandb_init(wandb_config: WandbConfig, run_config: dict | None = None):
    run = None
    if wandb_config.enable:
        run = wandb.init(
            project=wandb_config.project,
            name=wandb_config.name,
            notes=wandb_config.notes,
            group=wandb_config.group,
            tags=wandb_config.tags,
            config=run_config
        )

        global _wandb_log
        _wandb_log = wandb.log
    else:
        global logger
        logger.info("Wandb log is disabled. Using the local logger instead.")
    return run

logger = logging.getLogger("clacl")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def dummy_log(*args):
    logger.info("\n".join(repr(arg) for arg in args))

_wandb_log = dummy_log

def wandb_log(*args):
    _wandb_log(*args)

def wandb_log_table(name:str, columns=None, data=None, rows=None, dataframe=None, dtype=None):
    table = wandb.Table(columns=columns, data=data, rows=rows, dataframe=dataframe, dtype=dtype)
    _wandb_log({name: table})
