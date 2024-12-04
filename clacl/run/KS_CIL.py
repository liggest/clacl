from functools import cached_property

from clacl.task.KS_CIL import CLTask
from clacl.run.common import Pipeline

class KSPipeline(Pipeline):

    @cached_property
    def _task(self):
        return CLTask()

if __name__ == "__main__":
    KSPipeline().main()
