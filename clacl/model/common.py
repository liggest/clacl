
from torch.optim.lr_scheduler import ExponentialLR

class DummyScheduler(ExponentialLR):

    def __init__(self, optimizer, last_epoch: int = -1, verbose: bool = "deprecated"):
        # gamma = 1.0
        super().__init__(optimizer, 1.0, last_epoch, verbose)
