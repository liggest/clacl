from functools import cached_property
# import random

# from clacl.data.speech_commands import CLASSES
from clacl.task.KS_CL import CLTask, KSSubTask
from clacl.run.common import Pipeline

class KSPipeline(Pipeline):

    @cached_property
    def _task(self):
        # return CLTask([
        #     KSSubTask(CLASSES[  :15]),
        #     KSSubTask(CLASSES[15:18]),
        #     KSSubTask(CLASSES[18:21]),
        #     KSSubTask(CLASSES[21:24]),
        #     KSSubTask(CLASSES[24:27]),
        #     KSSubTask(CLASSES[27:30])
        # ])

        # random.seed(42)
        # random.shuffle(CLASSES)  # seed is not fixed at here
        CLASSES = [
            'tree', 'happy', 'nine', 'marvin', 'bird', 'on', 'right', 'bed', 'three', 'four', 
            'go', 'two', 'zero', 'eight', 'dog', 'sheila', 'no', 'wow', 'five', 'up', 
            'cat', 'house', 'left', 'six', 'off', 'stop', 'seven', 'yes', 'down', 'one'
        ]
        return CLTask(tasks = [
            KSSubTask(CLASSES[  :10]),
            KSSubTask(CLASSES[10:20]),
            KSSubTask(CLASSES[20:30])
        ])

if __name__ == "__main__":
    KSPipeline().main()
