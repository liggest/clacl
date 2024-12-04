from functools import cached_property

from clacl.task.finetune import FinetuningTask
from clacl.run.common import Pipeline

class FinetuningPipeline(Pipeline):

    @cached_property
    def _task(self):
        return FinetuningTask()

if __name__ == "__main__":
    FinetuningPipeline().main()
