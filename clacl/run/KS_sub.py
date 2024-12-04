from functools import cached_property

from clacl.task.KS_sub import KSSubTask
from clacl.run.common import Pipeline

class KSPipeline(Pipeline):

    @cached_property
    def _task(self):
        return KSSubTask()

if __name__ == "__main__":
    KSPipeline().main()
