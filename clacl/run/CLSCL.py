from functools import cached_property

from clacl.task.CLSCL import CLTask
from clacl.run.common import Pipeline

class CLSCLPipeline(Pipeline):

    @cached_property
    def _task(self):
        return CLTask()

    def _after_train(self):
        from pathlib import Path
        from clacl.model.wavlm_cl import cl_modules
        from clacl.util import logger
        base = Path("data/CLSCL/model")
        base.mkdir(exist_ok=True)
        pt_path = base / "cl_modules.pt"
        cl_modules.save(pt_path)
        logger.info(f"Saved cl_modules to {pt_path}")

if __name__ == "__main__":
    CLSCLPipeline().main()
