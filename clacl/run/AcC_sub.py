from functools import cached_property

from clacl.task.AcC_sub import AcCSubTask
from clacl.run.common import Pipeline

class AcCPipeline(Pipeline):

    @cached_property
    def _task(self):
        return AcCSubTask()

if __name__ == "__main__":
    AcCPipeline().main()
