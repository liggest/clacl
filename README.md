# Continual Learning for Audio CLassification (CLACL)

> Currently for study purpose and focusing on speech classification

## Installation
```
conda env create -f environment.yml
```

In some case, you may want to install PyTorch manually.
```
pip install torch torchaudio
```

## Running a task

Taking the task `CLSCL` and the config file `config_5.toml` as an example:

```
python -m clacl.run.CLSCL --config data/CLSCL/config_5.toml
```

To dump a default config file of the task:
```
python -m clacl.run.CLSCL --dump data/CLSCL/my_config.toml
```
