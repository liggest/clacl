
from pathlib import Path

from clacl.data.classification import Dataset as DatasetBase

CLASSES = [
    "ang", "hap", "exc", "sad", "fru", "fea", "sur", "neu", "xxx"
    # angry happy excited sad frustrated fearful surprised neutral ???
]

FIVE_CLASSES = ["ang", "hap", "exc", "sad", "neu"]

def parse_name(name: str):
    # Ses01F_impro01_M001 -> (Ses01F, impro01, ..., M001)
    session, script_or_impro, *_, last = name.split("_")
    session_id = session[3:-1].strip("0")  # Ses01F -> 1
    gender = last[0]  # M001 -> M
    utterance_id = last[1:]  # M001 -> 001
    return session_id, script_or_impro, gender, utterance_id

class Dataset(DatasetBase):
    
    BaseName = "IEMOCAP_full_release"

    label2id = {label: i for i, label in enumerate(CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    def __init__(self, 
                 data_path=f"data/iemocap/{BaseName}",
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
        s1 = self._data_path / "Session1"
        if not s1.is_dir():
            self._data_path /= self.BaseName
            s1 = self._data_path / "Session1"
        assert s1.is_dir()
