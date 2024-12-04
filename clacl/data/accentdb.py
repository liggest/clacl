
from pathlib import Path

import torch
from torchaudio.transforms import Resample

from clacl.data.common import SAMPLING_RATE
from clacl.data.classification import Dataset as DatasetBase

CLASSES = [
    "american",
    "australian",
    "bangla",
    "british",
    "indian",
    "malayalam",
    "odiya",
    "telugu",
    "welsh"
]

class Dataset(DatasetBase):

    label2id = {label: i for i, label in enumerate(CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    RESAMPLEs: dict[int, Resample] = {}

    def __init__(self, 
                 data_path="data/accentdb_extended",
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
        if self._data_path.stem != "data":
            self._data_path /= "data"
        assert self._data_path.is_dir()
    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # to single channel
        if waveform.size(0) > 1:  # may be 1 or 2
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample
        if sr != SAMPLING_RATE:
            RESAMPLE = self.RESAMPLEs.get(sr)
            if RESAMPLE is None:
                RESAMPLE = self.RESAMPLEs[sr] = Resample(sr, SAMPLING_RATE)  # sr may be 22050 or 48000
            waveform = RESAMPLE(waveform)
        waveform = super().preprocess(waveform, SAMPLING_RATE)  # asserts
        return waveform
