import logging
from src.util import Singleton


def get_logger(logging_level: str = None) -> logging.Logger:
    """
    Return the singleton DattaBot logger. If a logging_level is provided,
    update the logger's level. Otherwise, return it as-is.
    """
    logger_instance = DattaBotLogger.__call__()  # Singleton ensures only one
    if logging_level is not None:
        # Convert string to int if necessary (e.g., "DEBUG" -> 10)
        if isinstance(logging_level, str):
            logging_level = logging.getLevelName(logging_level)
        logger_instance.get_logger(logging_level)
    return logger_instance.get_logger()


class DattaBotLogger(object, metaclass=Singleton):
    _logger = None

    def __init__(
        self,
        logging_level: int = logging.INFO,
        logging_file_name: str = "dattabot.log",
    ) -> None:
        """
        CRITICAL = 50
        FATAL = CRITICAL
        ERROR = 40
        WARNING = 30
        WARN = WARNING
        INFO = 20
        DEBUG = 10
        NOTSET = 0
        """
        # Create a custom logger
        self._logger = logging.getLogger("dattabot")
        # Avoid duplicate handlers if re-initialized
        if not self._logger.hasHandlers():
            self._logger.setLevel(logging_level)
            # Create a file handler and set the logging level
            file_handler = logging.FileHandler(logging_file_name)
            file_handler.setLevel(logging_level)
            # Create a console handler and set the logging level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)
            # Create a formatter and attach it to the handlers
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            # Attach the handlers to the logger
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)
        self._logger.info(
            f"Logging level being used: {logging.getLevelName(logging_level)}"
        )

    def get_logger(self, logging_level: int = None):
        if logging_level is not None:
            self._logger.setLevel(logging_level)
        return self._logger
