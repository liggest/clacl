from functools import cached_property

from clacl.task.ER_sub import ERSubTask
from clacl.run.common import Pipeline

class ERPipeline(Pipeline):

    @cached_property
    def _task(self):
        return ERSubTask()

if __name__ == "__main__":
    ERPipeline().main()
