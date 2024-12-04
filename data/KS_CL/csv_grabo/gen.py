# Code partially derived from https://github.com/ga642381/SpeechPrompt-v2/tree/main/GSLM/preprocess/SCR_grabo
from typing import Iterable
from pathlib import Path
import sys

import pandas as pd


here = Path(__file__).parent  # csv/
# -> KS_CL/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig  # noqa: E402

# path to garbo/
data_path = GenCILConfig().data_path or root_path / "data/KS_CIL/grabo"

if (data_path / "grabo/speakers").is_dir():
    data_path = data_path / "grabo"

DataList = list[tuple[str, Path]]

def split_dataset(data_path: Path) -> tuple[DataList, DataList, DataList]:
    train_list, valid_list, test_list = [], [], []
    for speaker in (data_path / "speakers").iterdir():
        for entry in (speaker / "spchdatadir").iterdir():
            cnt = 0
            for audio_path in entry.glob("*.wav"):
                if cnt < 3:
                # if cnt < 2:
                    test_list.append((entry.name, audio_path))
                    # train_list.append((entry.name, audio_path))
                elif cnt < 6:
                    valid_list.append((entry.name, audio_path))
                    # valid_list.append((entry.name, audio_path))
                else:
                    train_list.append((entry.name, audio_path))
                    # test_list.append((entry.name, audio_path))
                cnt += 1
    return train_list, valid_list, test_list

train_list, valid_list, test_list = split_dataset(data_path)

def path_item(path: Path):
    return path.relative_to(data_path).as_posix()

def data_filter(data_list: list[tuple[str, Path]]):
    for label, path in data_list:
        yield path_item(path), label
    
def process(data_iter: Iterable[tuple[str, str]], file_name = "train.csv"):
    data = pd.DataFrame(data_iter, columns=["file", "label"])
    data.to_csv(here / file_name, index=False)

process(data_filter(train_list), "train.csv")
process(data_filter(valid_list), "valid.csv")
process(data_filter(test_list), "test.csv")
