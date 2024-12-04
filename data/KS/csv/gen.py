# Code partially derived from https://github.com/ga642381/SpeechPrompt-v2/tree/main/GSLM/preprocess/SCR_google_speech_commands

from typing import Iterable
from pathlib import Path
import re
import hashlib
import sys

import pandas as pd


here = Path(__file__).parent  # csv/
# -> KS/ -> data/ -> clacl/
root_path = here.parent.parent.parent

# -> KS/config.toml
config_path = here.parent / "config.toml"

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig  # noqa: E402

# path to speech_commands_v0.01
data_path = GenCILConfig().data_path or root_path / "data/KS/speech_commands_v0.01"
# path to speech_commands_test_set_v0.01
test_path = GenCILConfig().data_path or root_path / "data/KS/speech_commands_test_set_v0.01"

if config_path.is_file():
    # import sys
    # sys.path.append(root_path.as_posix())
    from clacl.task.KS import KSConfig
    import tomli

    with config_path.open("rb") as f:
        config = tomli.load(f)
        ks_config = KSConfig(**config["task"])
        # use paths in config.toml
        data_path = ks_config.data_path
        test_path = ks_config.test_path

CLASSES = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_unknown_", "_silence_"}

def split_dataset(
    data_path: Path, 
    max_uttr_per_class=2**27 - 1
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    train_list, valid_list = [], []

    for entry in data_path.iterdir():
        if not entry.is_dir() or entry.name == "_background_noise_":
            continue

        for audio_path in entry.glob("*.wav"):
            speaker_hashed = re.sub(r"_nohash_.*$", "", audio_path.name)
            hashed_again = hashlib.sha1(speaker_hashed.encode("utf-8")).hexdigest()
            percentage_hash = (int(hashed_again, 16) % (max_uttr_per_class + 1)) * (100.0 / max_uttr_per_class)

            if percentage_hash < 10:
                valid_list.append((entry.name, audio_path))
            elif percentage_hash < 20:
                pass  # testing set is discarded
            else:
                train_list.append((entry.name, audio_path))

    return train_list, valid_list

train_list, valid_list = split_dataset(data_path)

def path_item(path: Path):
    return Path(*path.parts[-2:]).as_posix()  # "xxx/yyy.wav"

def data_filter(data_list: list[tuple[str, Path]]):
    for label, path in data_list:
        if label in CLASSES:
            yield path_item(path), label
        else:
            yield path_item(path), "_unknown_"
    
    for path in (data_path / "_background_noise_").glob("*.wav"):
        yield path_item(path), "_silence_"

def test_gen():
    for dir_path in test_path.iterdir():
        if not dir_path.is_dir():
            continue
        for path in dir_path.glob("*.wav"):
            yield path_item(path), dir_path.name

def process(data_iter: Iterable[tuple[str, str]], file_name = "train.csv"):
    data = pd.DataFrame(data_iter, columns=["file", "label"])
    data.to_csv(here / file_name, index=False)

process(data_filter(train_list), "train.csv")
process(data_filter(valid_list), "valid.csv")
    
process(test_gen(), "test.csv")
    
