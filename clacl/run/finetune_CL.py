from functools import cached_property

from clacl.task.finetune_CL import FinetuningCLTask
from clacl.run.common import Pipeline

class FinetuningCLPipeline(Pipeline):

    @cached_property
    def _task(self):
        return FinetuningCLTask()

if __name__ == "__main__":
    FinetuningCLPipeline().main()
