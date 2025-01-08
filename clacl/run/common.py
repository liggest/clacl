
from functools import cached_property
from pathlib import Path

from clacl.task.common import WavMLClassificationTask as Task
from clacl.util import fix_seed, wandb_init, logger
from clacl.config import cli_config

class Pipeline:

    default_dump = Path("config_dump.toml")

    @cached_property
    def _task(self):
        return Task()
    
    def main(self):

        _cli = cli_config()

        seed = self._task.config.seed  # init task

        if _cli.seed is not None:
            seed = _cli.seed
            self._task.config.seed = seed  # override
            
        assert seed is not None
        fix_seed(seed)

        if dump_file := _cli.dump:
            self.dump(dump_file)

            import sys
            sys.exit(0)

        assert self._task._raw_config._config_file_exists()  # only start train if config is loaded from an existing file
        self.train()

    def dump(self, file: Path | bool | None = None):
        import tomli_w

        if file in (None, True, False):
            file = self.default_dump
        config_dict = self._task._raw_config.model_dump(mode = "json", exclude_none = True)  # None not exists in toml
        logger.info(repr(config_dict))

        with file.open("wb") as f:
            tomli_w.dump(config_dict, f)

        logger.info(f"Config dumped to {file.resolve().as_posix()}")

    def train(self):
        task = self._task
        config_dict = task.config.model_dump()
        
        wandb_init(task._raw_config.wandb, config_dict)
        
        logger.info(repr(config_dict))
        
        task.init_model()

        self._before_train()

        logger.info(repr(task.model.config))

        task.train()

        self._after_train()

    def _before_train(self):
        pass

    def _after_train(self):
        pass
