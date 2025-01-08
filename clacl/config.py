from typing import ClassVar
from pathlib import Path

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings import PydanticBaseSettingsSource, TomlConfigSettingsSource, YamlConfigSettingsSource

class CLIConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    config: Path | None = Field(None, validation_alias=AliasChoices("c", "config"))
    dump: bool | Path = Field(False, validation_alias=AliasChoices("d", "dump"))
    seed: int | None = None

_cli = None

def cli_config():
    global _cli
    if _cli is None:
        _cli = CLIConfig()
    return _cli

def file_config_base(path: Path):
    path = cli_config().config or path

    class FileConfig(BaseSettings):
        model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
            toml_file=path.with_suffix(".toml"), 
            yaml_file=[path.with_suffix(".yml"), path.with_suffix(".yaml")]
        )

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (TomlConfigSettingsSource(settings_cls), YamlConfigSettingsSource(settings_cls))
        
        @classmethod
        def _config_paths(cls):
            yield cls.model_config["toml_file"]
            yield from cls.model_config["yaml_file"]

        @classmethod
        def _config_file_exists(cls):
            return any(path.is_file() for path in cls._config_paths())

    return FileConfig

class GenCILConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    data_path: Path | None = Field(None, validation_alias=AliasChoices("d", "data", "data_path"))

# class WandbConfig(BaseModel):
#     enable: bool = True
#     project: str = "clacl"
#     name: str | None = None
#     notes: str | None = None
#     tags: list[str] | None = None
#     group: str | None = None

