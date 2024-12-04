from functools import cached_property

from clacl.task.IC import Task
from clacl.run.common import Pipeline

class ICPipeline(Pipeline):

    @cached_property
    def _task(self):
        return Task()
    
    def _after_train(self):
        print(repr(self._task.weight_result()))

if __name__ == "__main__":
    ICPipeline().main()
