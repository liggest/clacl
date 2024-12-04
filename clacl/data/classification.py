
from pathlib import Path

import torch
from torch.utils.data import Dataset as DatasetBase
import torchaudio
import pandas as pd

from clacl.data.common import SAMPLING_RATE, DataPiece, PLabelConfig

class Dataset(DatasetBase):
    
    label2id: dict[str, int] = {}
    id2label: dict[int, str] = {}

    def __init__(self, 
                 data_path="data/dataset", 
                 csv_file="train.csv", 
                 classes: list[str] | None = None, 
                 label_config: PLabelConfig | None = None):
        super().__init__()
        self.data_path = Path(data_path)
        self.df = self.raw_df = self.from_csv(csv_file)
        self.classes = classes
        if label_config:
            self.label2id, self.id2label = label_config.label2id, label_config.id2label
    
    @property
    def data_path(self):
        return self._data_path
    
    @data_path.setter
    def data_path(self, val: Path):
        self._data_path = val
        assert self._data_path.is_dir()
    
    @property
    def classes(self):
        return self._classes
    
    @classes.setter
    def classes(self, val: list[str] | None):
        self._classes = val
        if self._classes:
            # filter df with self._classes
            self.df = self.raw_df[self.raw_df["label"].isin(self._classes)].reset_index(drop=True)

    def from_csv(self, file: Path):
        df = pd.read_csv(file)
        df["file"] = self.data_path / df["file"]  # to full path
        return df
    
    def load_audio(self, wav_path: Path):
        return torchaudio.load(wav_path)

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        assert sr == SAMPLING_RATE, f"wrong sampling rate: {sr}, should be {SAMPLING_RATE}"
        assert waveform.size(0) == 1, f"waveform should be single channel, current size: {waveform.size()}"
        return waveform
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> DataPiece:
        wav_path, label = self.df.loc[idx]
        waveform = self.preprocess(*self.load_audio(wav_path))
        class_id = self.label2id[label]
        return {"file": wav_path, "array": waveform, "label": class_id}
