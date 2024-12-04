from functools import cached_property

from clacl.task.KS import Task
from clacl.run.common import Pipeline

class KSPipeline(Pipeline):

    @cached_property
    def _task(self):
        return Task()
    
    # def _after_train(self):
    #     print(repr(self._task.weight_result()))

if __name__ == "__main__":
    KSPipeline().main()
