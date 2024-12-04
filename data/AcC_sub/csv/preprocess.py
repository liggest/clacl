from pathlib import Path
import sys

import torchaudio
from torchaudio.transforms import Resample

SAMPLING_RATE = 16000

resamples: dict[int, Resample] = {}

here = Path(__file__).parent  # csv/
# -> AcC_sub/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig # noqa: E402

base_name = "accentdb_extended"
target_name = "accentdb_extended_processed"

data_path = GenCILConfig().data_path or root_path / f"data/AcC_sub/{base_name}/"
parent_path = data_path.parent  # ../accentdb_extended

data_path = data_path / "data"
assert data_path.is_dir()

target_path = parent_path / target_name / "data"  # accentdb_extended_processed/data
target_path.mkdir(parents=True, exist_ok=True)

def preprocess(wav_path: Path, target_path: Path):
    target_wav_path = target_path / wav_path.name
    if target_wav_path.exists():
        print("Skipped:", target_wav_path.as_posix())
        return
    
    waveform, sr = torchaudio.load(wav_path)
    RESAMPLE = resamples.get(sr)
    if RESAMPLE is None:
        RESAMPLE = resamples[sr] = Resample(sr, SAMPLING_RATE)  # sr may be 22050 or 48000
    if waveform.size(0) > 1:  # may be 1 or 2
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = RESAMPLE(waveform)

    torchaudio.save(target_wav_path, waveform, SAMPLING_RATE)
    print("Processed:", target_wav_path.as_posix())

for language in data_path.iterdir():
    for speaker in language.iterdir():
        current = target_path / language.name / speaker.name
        current.mkdir(parents=True, exist_ok=True)
    
        for wav_path in speaker.glob("*.wav"):
            preprocess(wav_path, current)
