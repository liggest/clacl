from functools import cached_property

from clacl.task.LID_sub import LIDSubTask
from clacl.run.common import Pipeline

class LIDPipeline(Pipeline):

    @cached_property
    def _task(self):
        return LIDSubTask()

if __name__ == "__main__":
    LIDPipeline().main()
