from functools import cached_property

from clacl.task.IC_sub import ICSubTask
from clacl.run.common import Pipeline

class ICPipeline(Pipeline):

    @cached_property
    def _task(self):
        return ICSubTask()

if __name__ == "__main__":
    ICPipeline().main()
