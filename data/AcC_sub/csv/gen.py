# Code partially derived from https://github.com/ga642381/SpeechPrompt-v2/blob/main/GSLM/preprocess/AcC_accentdb/dataset.py

from typing import Iterable
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

here = Path(__file__).parent  # csv/
# -> AcC_sub/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig as BaseCILConfig # noqa: E402

class GenCILConfig(BaseCILConfig):
    seed: int = 42

config = GenCILConfig()
data_path = config.data_path or root_path / "data/AcC_sub/accentdb_extended/"

# path to accentdb_extended/data
data_path = data_path / "data"
assert data_path.exists()

def split_dataset(data_path: Path, seed: int = 42):
    audios, labels = [], []
    for language in data_path.iterdir():
        for speaker in language.iterdir():
            audios.extend(speaker.glob("*.wav"))
            labels.extend([language.stem] * (len(audios) - len(labels)))
    
    train_audios, test_audios, train_labels, test_labels = train_test_split(audios, labels, test_size=0.2, random_state=seed)
    train_audios, valid_audios, train_labels, valid_labels = train_test_split(train_audios, train_labels, test_size=0.2, random_state=seed)
    train_split, valid_split, test_split = zip(train_labels, train_audios), zip(valid_labels, valid_audios), zip(test_labels, test_audios)
    return train_split, valid_split, test_split

def path_item(path: Path):
    return Path(*path.parts[-3:]).as_posix()  # "language/speaker/xxx.wav"

def data_filter(data_list: Iterable[tuple[str, Path]]):
    for label, path in data_list:
        yield path_item(path), label
def process(data_iter: Iterable[tuple[str, str]], file_name = "train.csv"):
    data = pd.DataFrame(data_iter, columns=["file", "label"])
    data.to_csv(here / file_name, index=False)

train_split, valid_split, test_split = split_dataset(data_path, seed=config.seed)
process(data_filter(train_split), "train.csv")
process(data_filter(valid_split), "valid.csv")
process(data_filter(test_split), "test.csv")
