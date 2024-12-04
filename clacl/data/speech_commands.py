# Code partially derived from https://github.com/huangyz0918/kws-continual-learning/blob/master/model/dataloader/speech_loader.py and https://github.com/ga642381/SpeechPrompt-v2/blob/main/GSLM/preprocess/SCR_google_speech_commands/dataset.py

import random

import torch

from clacl.data.common import GAIN, DataPiece
from clacl.data.classification import Dataset as DatasetBase

SAMPLE_LENGTH = 16000

CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",    # 10
    # "_unknown_",
    # "_silence_",
    # SUPERB
    "nine", "three", "bed", "wow", "happy",  # 15
    "four", "dog", "cat",     # 18
    "five", "tree", "one",    # 21
    "eight", "bird", "seven", # 24
    "six", "two", "marvin",   # 27
    "sheila", "house", "zero" # 30
]

SUPERB_CLASSES = [*CLASSES[:10], "_unknown_", "_silence_"]

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]

class Dataset(DatasetBase):

    label2id = {label: i for i, label in enumerate(CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    @classmethod
    def of_classes(cls, classes: list[str]):
        if "_silence_" in classes:
            return DatasetWithSilence
        return cls

    def __init__(self, 
                 data_path="data/speech_commands_v0.01",
                 csv_file="train.csv", 
                 classes=None, 
                 label_config=None):
        super().__init__(data_path, csv_file, classes, label_config)

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        waveform = super().preprocess(waveform, sr)
        waveform = GAIN(waveform)
        return waveform

    @property
    def sample_weights(self):
        count = self.df.value_counts("label")
        total = len(self)
        return (total / count[self.df["label"]]).to_list()

class DatasetWithSilence(Dataset):

    def __getitem__(self, idx) -> DataPiece:
        item = super().__getitem__(idx)

        if item["label"] == self.label2id["_silence_"]:  
            # randomly sample 1s _silence_
            waveform = item["array"]
            offset = random.randint(0, waveform.shape[1] - SAMPLE_LENGTH)
            # wav = wav[offset : offset + SAMPLE_LENGTH]
            item["array"] = waveform.narrow(1, offset, SAMPLE_LENGTH)
        return item
