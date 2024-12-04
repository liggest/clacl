# obtain .csv files from https://github.com/huangyz0918/kws-continual-learning/tree/master/dataset/splits

from pathlib import Path

import pandas as pd

here = Path(__file__).parent  # csv/
# -> KS/ -> data/ -> clacl/
root_path = here.parent.parent.parent

# -> clacl/ -> ../kws-continual-learning/dataset/splits/
txt_path = root_path.parent / "kws-continual-learning/dataset/splits"

def gen_rows(file: Path):
    with file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # "label/file_name.wav"
            label, file_name = line.split("/")
            yield line, label

def process(file: Path):
    data = pd.DataFrame(gen_rows(file), columns=["file", "label"])
    data.to_csv(here / file.with_suffix(".csv").name, index=False)

for file in txt_path.iterdir():
    if file.suffix == ".txt":
        process(file)
