# Code partially derived from https://github.com/ga642381/SpeechPrompt-v2/blob/main/GSLM/preprocess/LID_voxforge/dataset.py

from typing import Iterable
import random
from pathlib import Path
import sys
from itertools import islice

import pandas as pd

here = Path(__file__).parent  # csv/
# -> LID_sub/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig as BaseCILConfig  # noqa: E402

class GenCILConfig(BaseCILConfig):
    seed: int = 42
    train_num: int = 1200
    valid_num: int = 300
    test_num: int = 300
    # for one lang

base_name = "VoxForge"
config = GenCILConfig()
data_path = config.data_path or root_path / f"data/LID_sub/{base_name}/"

# .tgz files are extracted to data/ with tgz.py
# path to VoxForge/data
if not (data_path / "data").exists():
    data_path = data_path / base_name
assert (data_path / "data").exists()
def wav_path_gen(lang_dir: Path):
    for items in lang_dir.iterdir():
        yield from (items / "wav").glob("*.wav")
def split_dataset(data_path: Path, seed: int = 42, train_num: int = 1200, valid_num: int = 300, test_num: int = 300):
    train_split, valid_split, test_split = [], [], []
    
    random.seed(seed)
    valid_end = train_num + valid_num
    test_end = valid_end + test_num

    for lang_dir in data_path.iterdir():
        lang = lang_dir.name
        
        wavs = list(islice(wav_path_gen(lang_dir), test_end))
        random.shuffle(wavs)

        train_split.extend((lang, wav) for wav in wavs[:train_num])
        valid_split.extend((lang, wav) for wav in wavs[train_num:valid_end])
        test_split.extend((lang, wav) for wav in wavs[valid_end:test_end])
    
    return train_split, valid_split, test_split

def path_item(path: Path):
    return Path(*path.parts[-4:]).as_posix()  # "de/xxx/wav/yyy.wav"

def data_filter(data_list: Iterable[tuple[str, Path]]):
    for label, path in data_list:
        yield path_item(path), label

def process(data_iter: Iterable[tuple[str, str]], file_name = "train.csv"):
    data = pd.DataFrame(data_iter, columns=["file", "label"])
    data.to_csv(here / file_name, index=False)

train_split, valid_split, test_split = split_dataset(data_path / "data", 
    seed=config.seed, train_num=config.train_num, valid_num=config.valid_num, test_num=config.test_num
)
process(data_filter(train_split), "train.csv")
process(data_filter(valid_split), "valid.csv")
process(data_filter(test_split), "test.csv")


