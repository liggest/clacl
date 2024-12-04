
from functools import cached_property
from dataclasses import dataclass
from typing import TypedDict, Callable, Protocol
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from torchaudio.transforms import Vol


SAMPLING_RATE = 16000
GAIN = Vol(gain=-3.0, gain_type='db')

class DataPiece(TypedDict):
    file: str
    array: torch.Tensor
    label: int | list[int]

class DataPieceCollator:

    sampling_rate = SAMPLING_RATE
    
    def __init__(self, extractor: Wav2Vec2FeatureExtractor, sampling_rate: int | None = None, lthresh: int | None = None):
        self.lthresh = lthresh
        self.extractor = extractor
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
        
    def __call__(self, batch: list[DataPiece]):
        waveforms, targets = [], []
        for data in batch:
            # if self.lthresh is None:
            #     waveforms += [data['array'].numpy().flatten()]
            # else:
            #     waveforms += [data['array'].numpy().flatten()[:self.lthresh]]
            # targets += [data['label']]
            array = data["array"].numpy().flatten()
            # print(array.shape, data["file"])
            if self.lthresh is not None:
                array = array[:self.lthresh]
            waveforms.append(array)
            targets.append(data["label"])
        targets = torch.LongTensor(targets)
        inputs = self.extractor(waveforms, sampling_rate=self.sampling_rate, padding=True, return_tensors="pt")

        return inputs, targets


class PDataLoaders(Protocol):
    train: DataLoader | None
    valid: DataLoader | None
    test:  DataLoader | None

@dataclass
class DataLoaders:
    train: DataLoader | None
    valid: DataLoader | None
    test:  DataLoader | None


class LazyDataLoaders:
    def __init__(self, 
        get_train:  Callable[[], DataLoader | None], 
        get_valid:  Callable[[], DataLoader | None], 
        get_test :  Callable[[], DataLoader | None]
    ):
        self._get_train = get_train
        self._get_valid = get_valid
        self._get_test = get_test

    @cached_property
    def train(self):
        return self._get_train()
    
    @cached_property
    def valid(self):
        return self._get_valid()
    
    @cached_property
    def test(self):
        return self._get_test()

class PLabelConfig(Protocol):
    label2id: dict[str, int]
    id2label: dict[int, str]

@dataclass
class LabelConfig:
    label2id: dict[str, int]
    id2label: dict[int, str]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
