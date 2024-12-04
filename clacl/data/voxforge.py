
from pathlib import Path

from clacl.data.classification import Dataset as DatasetBase

CLASSES = ["de", "en", "es", "fr", "it", "ru"]

class Dataset(DatasetBase):

    label2id = {label: i for i, label in enumerate(CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    BaseName = "data"

    def __init__(self, 
                 data_path=f"data/VoxForge/{BaseName}",
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
        s1 = self._data_path / "en"
        if not s1.is_dir():
            self._data_path /= self.BaseName
            s1 = self._data_path / "en"
        assert s1.is_dir()
