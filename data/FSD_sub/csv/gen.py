# Code partially derived from https://github.com/ga642381/SpeechPrompt-v2/blob/main/GSLM/preprocess/FSD_asvspoof/preprocessor.py

from typing import Literal, Iterable
from pathlib import Path
import sys

import pandas as pd

here = Path(__file__).parent  # csv/
# -> FSD_sub/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig # noqa: E402

config = GenCILConfig()
data_path = config.data_path or root_path / "data/FSD_sub/LA/"

assert data_path.exists()

cm_path = data_path / "ASVspoof2019_LA_cm_protocols"
def gen_rows(cm_path: Path, split: Literal["train", "dev", "eval"] = "train"):
    trl_trn = "trn" if split == "train" else "trl"
    cm_file = cm_path / f"ASVspoof2019.LA.cm.{split}.{trl_trn}.txt"
    with cm_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # "LA_0069 LA_D_1047731 - - bonafide"
            _, file_name, *_, label = line.split()
            file_name = f"ASVspoof2019_LA_{split}/flac/{file_name}.flac"
            yield file_name, label
def process(data_iter: Iterable[tuple[str, str]], file_name = "train.csv"):
    data = pd.DataFrame(data_iter, columns=["file", "label"])
    data.to_csv(here / file_name, index=False)
    # count amount of each label
    counts = data["label"].value_counts()
    print("**", file_name)
    print(counts)
    # print(counts / float(counts.sum()))
    print(counts / counts.sum())
    breakpoint()

process(gen_rows(cm_path, "train"), "train.csv")
process(gen_rows(cm_path, "dev"), "valid.csv")
process(gen_rows(cm_path, "eval"), "test.csv")
