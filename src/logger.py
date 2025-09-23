import logging

from src.util import Singleton

try:
    import torch.distributed as dist
except ImportError:
    dist = None


class RankFilter(logging.Filter):
    """
    Filter to only log on GPU with rank 0.
    """

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "all_ranks") and record.all_ranks:
            return True
        return self.rank == 0


class DattaBotLoggerWrapper:
    """
    Wrapper around standard logger to add `all_ranks` kwarg support in log calls.
    """

    def __init__(self, logger: logging.Logger, rank: int = 0):
        self._logger = logger
        self.rank = rank

    def debug(self, msg, *args, all_ranks=False, **kwargs):
        self._logger.debug(msg, *args, extra={"all_ranks": all_ranks}, **kwargs)

    def info(self, msg, *args, all_ranks=False, **kwargs):
        self._logger.info(msg, *args, extra={"all_ranks": all_ranks}, **kwargs)

    def warning(self, msg, *args, all_ranks=False, **kwargs):
        self._logger.warning(msg, *args, extra={"all_ranks": all_ranks}, **kwargs)

    def error(self, msg, *args, all_ranks=False, **kwargs):
        self._logger.error(msg, *args, extra={"all_ranks": all_ranks}, **kwargs)

    def critical(self, msg, *args, all_ranks=False, **kwargs):
        self._logger.critical(msg, *args, extra={"all_ranks": all_ranks}, **kwargs)

    def exception(self, msg, *args, all_ranks=False, exc_info=True, **kwargs):
        self._logger.error(
            msg, *args, exc_info=exc_info, extra={"all_ranks": all_ranks}, **kwargs
        )

    def __getattr__(self, attr):
        # Delegate other attributes/methods to underlying logger
        return getattr(self._logger, attr)


class DattaBotLogger(object, metaclass=Singleton):
    _logger = None

    def __init__(
        self,
        logging_level: int = logging.INFO,
        logging_file_name: str = "dattabot.log",
        log_all_ranks: bool = False,
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
        if self._logger is not None:
            # Already initialized
            return
        assert isinstance(
            logging_level, int
        ), f"Logging level must be an int, logging level passed in {logging_level}"

        # Determine rank for filtering (default 0)
        self.rank = 0
        self.log_all_ranks = log_all_ranks
        if dist is not None and dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()

        logger = logging.getLogger("dattabot")
        if not logger.hasHandlers():
            logger.setLevel(logging_level)

            file_handler = logging.FileHandler(logging_file_name)
            file_handler.setLevel(logging_level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)

            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            if not self.log_all_ranks:
                rank_filter = RankFilter(self.rank)
                file_handler.addFilter(rank_filter)
                console_handler.addFilter(rank_filter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        self._logger = DattaBotLoggerWrapper(logger, rank=self.rank)
        self._logger.info(
            f"Logging level being used: {logging.getLevelName(logging_level)}"
        )

    def _update_filters(self):
        for handler in self._logger.handlers:
            handler.filters.clear()
            if not self.log_all_ranks:
                handler.addFilter(RankFilter(self.rank))

    def get_logger(self):
        return self._logger

    def set_log_level(self, level: int):
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)
        self._logger.info(
            f"Logging level changed to: {logging.getLevelName(logging.getLevelName(level))}"
        )

    def set_log_all_ranks(self, enabled: bool):
        self.log_all_ranks = enabled
        self._update_filters()
        self._logger.info(f"log_all_ranks set to: {enabled}")


_logger_singleton: DattaBotLogger = None


def get_logger(
    logging_level: int = None, log_all_ranks: bool = False
) -> DattaBotLoggerWrapper:
    """
    Return the singleton logger. If the logger already exists and a new
    logging level or log_all_ranks setting is passed in, it updates them.
    """
    global _logger_singleton

    if _logger_singleton is None:
        _logger_singleton = DattaBotLogger(
            logging_level=logging_level or logging.INFO,
            log_all_ranks=log_all_ranks,
        )
    else:
        # Update log level if needed.
        if logging_level is not None:
            assert isinstance(
                logging_level, int
            ), f"Logging level must be an int, logging level passed in {logging_level}"
            current_level = _logger_singleton._logger.level
            if logging_level != current_level:
                _logger_singleton.set_log_level(logging_level)
        # Update filter if needed.
        if log_all_ranks != _logger_singleton.log_all_ranks:
            _logger_singleton.set_log_all_ranks(log_all_ranks)

    return _logger_singleton.get_logger()
