from functools import cached_property

from clacl.task.FSD_sub import FSDSubTask
from clacl.run.common import Pipeline

class FSDPipeline(Pipeline):

    @cached_property
    def _task(self):
        return FSDSubTask()

if __name__ == "__main__":
    FSDPipeline().main()
