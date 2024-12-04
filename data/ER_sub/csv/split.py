
from typing import Literal
from pathlib import Path
import sys

from pydantic import Field, AliasChoices
import pandas as pd

here = Path(__file__).parent  # csv/
# -> ER_sub/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())

from clacl.config import GenCILConfig  # noqa: E402

class SplitCILConfig(GenCILConfig):
    test_session: Literal[1, 2, 3, 4, 5] = Field(default=5, validation_alias=AliasChoices("test_session", "session", "fold"))
    valid_ratio: float = Field(default=0.2, validation_alias=AliasChoices("valid_ratio", "valid"))
    seed: int = 42

config = SplitCILConfig()
csv_path = config.data_path or here / "data.csv"

def split_csv(csv_path: Path, test_session: Literal[1, 2, 3, 4, 5] = 5, valid_ratio: float = 0.2, seed: int = 42):
    data = pd.read_csv(csv_path)
    test_prefix = f"Session{test_session}"
    test_mask = data["file"].str.startswith(test_prefix)
    train_valid_split, test_split = data[~test_mask], data[test_mask]
    train_split, valid_split = None, None
    for lable, grouped in train_valid_split.groupby("label"):
        print(f"Splitting label - {lable}")
        one_train_split, one_valid_split = split_valid(grouped, valid_ratio, seed)
        if train_split is None:
            train_split = one_train_split
        else:
            train_split = pd.concat([train_split, one_train_split])
        if valid_split is None:
            valid_split = one_valid_split
        else:
            valid_split = pd.concat([valid_split, one_valid_split])
    train_split = train_split.sort_values("file")
    valid_split = valid_split.sort_values("file")
    return train_split, valid_split, test_split
def split_valid(data: pd.DataFrame, valid_ratio: float = 0.2, seed: int = 42):
    # shuffel data
    data = data.sample(frac=1, random_state=seed)
    train_len = int(len(data) * (1 - valid_ratio))
    return data.iloc[:train_len], data.iloc[train_len:]
def save(data: pd.DataFrame, file_name: str):
    data.to_csv(here / file_name, index=False)

train_split, valid_split, test_split = split_csv(csv_path, test_session=config.test_session, valid_ratio=config.valid_ratio, seed=config.seed)
save(train_split, "train.csv")
save(valid_split, "valid.csv")
save(test_split, "test.csv")
