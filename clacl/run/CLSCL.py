from functools import cached_property

from clacl.task.CLSCL import CLTask
from clacl.run.common import Pipeline

class CLSCLPipeline(Pipeline):

    @cached_property
    def _task(self):
        return CLTask()

if __name__ == "__main__":
    CLSCLPipeline().main()
