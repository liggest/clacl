
from clacl.data.classification import Dataset as DatasetBase

CLASSES = [
    "bonafide",
    "spoof"
]

class Dataset(DatasetBase):

    label2id = {label: i for i, label in enumerate(CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    def __init__(self, 
                 data_path="data/ASVspoof2019/LA",
                 csv_file="train.csv", 
                 classes=None, 
                 label_config=None):
        super().__init__(data_path, csv_file, classes, label_config)

    @property
    def label_weights(self):
        counts = self.df["label"].value_counts()
        weight = 1 / counts
        return weight / weight.sum()
