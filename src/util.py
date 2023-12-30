import logging

TOKENIZER_MODEL_PATH: str = "tokenizer.model"


def get_default_logger() -> logging.Logger:
    return DattaBotLogger.__call__().get_logger()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DattaBotLogger(object, metaclass=Singleton):
    _logger = None

    def __init__(
        self, logging_level: int = logging.INFO, logging_file_name: str = "dattabot.log"
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
        # # Create a custom logger
        self._logger = logging.getLogger("dattabot")
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

    def get_logger(self):
        return self._logger
