import os
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from src.util import Singleton
from src.logger import get_logger

ABSOLUTE_PATH_CONFIG_DIR = os.path.join(os.getcwd(), "configs")
CONFIG_FILE_NAME = "dattabot_v1"


def get_agent_config() -> DictConfig:
    return AgentConfig.__call__().get_config()


class AgentConfig(object, metaclass=Singleton):
    _config = None

    def __init__(self) -> None:
        self._version_base = None
        self._config_dir = ABSOLUTE_PATH_CONFIG_DIR
        self._config_file_name = CONFIG_FILE_NAME
        initialize_config_dir(
            version_base=self._version_base,
            config_dir=self._config_dir,
            job_name=self._config_file_name,
        )
        self._config = compose(config_name=self._config_file_name)
        _logger = get_logger(logging_level=self._config.env.logging_level)
        _logger.info(f"Using agent config file: {self._config_file_name}")
        _logger.debug(f"Contents of agent config file:\n{self.get_config_str()}")

    def get_config(self) -> DictConfig:
        return self._config

    def get_config_str(self) -> str:
        return OmegaConf.to_yaml(self.get_config())
