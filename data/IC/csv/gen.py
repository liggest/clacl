# obtain .csv files from https://github.com/sinhat98/adapter-wavlm/tree/main/IC

from pathlib import Path
import pandas as pd

here = Path(__file__).parent  # csv/
# -> IC/ -> data/ -> clacl/
root_path = here.parent.parent.parent

# -> clacl/ -> ../adapter-wavlm/IC/
csv_path = root_path.parent / "adapter-wavlm/IC/"

def no_path_head(path: str):
    return path.removeprefix("../data/fsc/")

def process(csv: Path):
    name = csv.name
    data = pd.read_csv(csv)
    data = data.drop(columns=["ID", "duration", "spk_id", "semantics", "transcript"])
    data["wav"] = data["wav"].apply(no_path_head)
    data.to_csv(here / name, index=False)

for file in csv_path.iterdir():
    if file.suffix == ".csv":
        process(file)
