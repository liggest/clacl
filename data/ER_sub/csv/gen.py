# Code partially derived from https://github.com/sinhat98/adapter-wavlm/blob/main/SER/utils.py

from typing import Iterable
from pathlib import Path
import sys

import pandas as pd

here = Path(__file__).parent  # csv/
# -> ER_sub/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig  # noqa: E402

base_name = "IEMOCAP_full_release"
data_path = GenCILConfig().data_path or root_path / "data/ER_sub/iemocap/"

# path to IEMOCAP_full_release/Session1
if not (data_path / "Session1").exists():
    data_path = data_path / base_name

assert (data_path / "Session1").exists()

def parse_file(file: Path):
    with open(file, "r", encoding="utf-8") as f:
        # extract data from line like "[46.5800 - 52.1900]	Ses01F_impro01_F007	fru	[2.5000, 3.5000, 3.5000]"
        # with template "[start - end]	file	emotion	[activation, valence, dominance]"
        for line in f:
            if not line.startswith("["):
                continue
            # lines like: "[46.5800 - 52.1900]    Ses01F_impro01_F007    fru    [2.5000, 3.5000, 3.5000]"
            interval, name, emotion, scores = line.strip().split("\t")
            # "[46.5800 - 52.1900]" -> "46.5800 - 52.1900" -> ["46.5800", "52.1900"]
            # start_time, end_time = interval.strip("[]").split(" - ")
            # "[2.5000, 3.5000, 3.5000]" -> "2.5000, 3.5000, 3.5000" -> ["2.5000", "3.5000", "3.5000"]
            # activation, valence, dominance = scores.strip("[]").split(", ")
            yield name, emotion

def extract(data_path: Path):
    # for each session
    for i in range(1, 6):
        session_path = data_path / f"Session{i}" / "dialog" / "EmoEvaluation"
        for file in session_path.glob("[!.]*.txt"):
            for name, emotion in parse_file(file):
                # Ses01F_impro01_M001 -> Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_M001.wav
                wav_path = f"Session{i}/sentences/wav/{name[:-5]}/{name}.wav"
                yield wav_path, emotion

def process(data_iter: Iterable[tuple[str, str]], file_name = "data.csv"):
    data = pd.DataFrame(data_iter, columns=["file", "label"])
    data.to_csv(here / file_name, index=False)

process(extract(data_path), "data.csv")
