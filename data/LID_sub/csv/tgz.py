
from pathlib import Path
import sys
import tarfile
from functools import partial
from concurrent.futures import ProcessPoolExecutor

here = Path(__file__).parent  # csv/
# -> LID_sub/ -> data/ -> clacl/
root_path = here.parent.parent.parent

sys.path.append(root_path.as_posix())
from clacl.config import GenCILConfig # noqa: E402

base_name = "VoxForge"
data_path = GenCILConfig().data_path or root_path / f"data/LID_sub/{base_name}/"

# path to VoxForge/downloads
if not (data_path / "downloads").exists():
    data_path = data_path / base_name
assert (data_path / "downloads").exists()
def extract_tgz(tgz_path: Path, target_dir: Path):
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print(f"Extracted: {tgz_path} -> {target_dir}")
    except Exception as e:
        print(f"Failed: {tgz_path}, Exception: {e!r}")

def extract_tgz_multi(data_path: Path):
    downloads_dir = data_path / "downloads"
    target_dir = data_path / "data"

    with ProcessPoolExecutor() as executor:
        for lang_dir in downloads_dir.iterdir():
            target_lang = target_dir / lang_dir.name
            target_lang.mkdir(parents=True, exist_ok=True)
            
            tgz_dir = lang_dir / "Trunk/Audio/Main/16kHz_16bit"

            executor.map(partial(extract_tgz, target_dir=target_lang), tgz_dir.iterdir())
    
extract_tgz_multi(data_path)
