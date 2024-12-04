from pathlib import Path

import torch

from clacl.data.common import GAIN
from clacl.data.classification import Dataset as DatasetBase

CLASSES = [f"recording{i}" for i in range(1, 37)]

class Dataset(DatasetBase):

    BaseName = "grabo"

    label2id = {label: i for i, label in enumerate(CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    def __init__(self, 
                 data_path=f"data/{BaseName}",
                 csv_file="train.csv", 
                 classes=None, 
                 label_config=None):
        super().__init__(data_path, csv_file, classes, label_config)

    @property
    def data_path(self):
        return self._data_path
    
    @data_path.setter
    def data_path(self, val: Path):
        self._data_path = val
        speakers = self._data_path / "speakers"
        if not speakers.is_dir():
            self._data_path /= self.BaseName
            speakers = self._data_path / "speakers"
        assert speakers.is_dir()

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        waveform = super().preprocess(waveform, sr)
        waveform = GAIN(waveform)
        return waveform
