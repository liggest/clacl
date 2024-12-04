# Code partially derived from https://github.com/sinhat98/adapter-wavlm/blob/main/IC/utils.py

from pathlib import Path

from torch.utils.data import Dataset as DatasetBase
import torchaudio
import pandas as pd

from clacl.data.common import SAMPLING_RATE, DataPiece, PLabelConfig

CLASSES = [
    # 0 - 5
    "change language", "activate", "deactivate", "increase", "decrease", "bring",
    # 6 - 19
    "none_object", "music", "lights", "volume", "heat", "lamp", "newspaper",
    "juice", "socks", "Chinese", "Korean", "English", "German", "shoes",
    # 20 - 23
    "none_location", "kitchen", "bedroom", "washroom"
]

class Dataset(DatasetBase):

    ACT_STOP, OBJ_STOP, LOC_STOP = 6, 20, 24

    label2id = {label: i for i, label in enumerate(CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    def __init__(self, config: PLabelConfig | None = None, data_path="data/fsc", csv_file="train.csv"):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.data_path = Path(data_path) # ./data/fsc
        assert self.data_path.is_dir()
        # config = WavLMConfig.from_pretrained("superb/wav2vec2-large-superb-ic")
        # config = External().pre_config
        if config:
            self.label2id, self.id2label = config.label2id, config.id2label
        # self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-ic")
        # self.extractor = External().extractor
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> DataPiece:
        wav_path = self.df.loc[idx, 'wav']
        array, sr = torchaudio.load(self.data_path / wav_path)
        assert sr == SAMPLING_RATE, "wrong sampling rate"
        # label = []

        # for slot in ["action", "object", "location"]:
        #     value = self.df.loc[idx][slot]
        #     label.append(self.label2id[value])
        label = [self.label2id[label] for label in self.df.loc[idx, ["action", "object", "location"]]]

        return {'file': wav_path, 'array': array, 'label': label}
