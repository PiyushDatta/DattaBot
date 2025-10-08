import logging
from time import gmtime

from src.util import Singleton


def _get_current_rank() -> int:
    """
    Dynamically get the current distributed rank.
    This is called every time we need to check the rank, not just once.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except (ImportError, RuntimeError):
        pass
    return 0


class RankFilter(logging.Filter):
    """
    Filter to only log on GPU with rank 0.
    Uses dynamic rank detection instead of cached value.
    """

    def __init__(self):
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "all_ranks") and record.all_ranks:
            return True
        # Dynamically check rank each time
        return _get_current_rank() == 0


class DattaBotLoggerWrapper:
    """
    Wrapper around standard logger to add `all_ranks` kwarg support in log calls.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    @property
    def rank(self):
        """Dynamically get current rank."""
        return _get_current_rank()

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

        # Store this for later use
        self.log_all_ranks = log_all_ranks

        logger = logging.getLogger("dattabot")
        if not logger.hasHandlers():
            logger.setLevel(logging_level)
            file_handler = logging.FileHandler(logging_file_name)
            file_handler.setLevel(logging_level)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)
            formatter = logging.Formatter(
                "%(asctime)s UTC - %(levelname)s - %(filename)s - %(name)s.%(funcName)s - %(message)s",
                # 12-hour format with AM/PM
                datefmt="%I:%M:%S %p",
            )
            # Use UTC time
            formatter.converter = gmtime
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            if not self.log_all_ranks:
                # Use dynamic rank filter (no rank parameter needed)
                rank_filter = RankFilter()
                file_handler.addFilter(rank_filter)
                console_handler.addFilter(rank_filter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        # No rank parameter needed - it's dynamic
        self._logger = DattaBotLoggerWrapper(logger)

        # Only log initialization on rank 0
        if _get_current_rank() == 0:
            self._logger.info(
                f"Logging level being used: {logging.getLevelName(logging_level)}"
            )

    def _update_filters(self):
        """Update filters on all handlers."""
        for handler in self._logger._logger.handlers:
            handler.filters.clear()
            if not self.log_all_ranks:
                # Re-add dynamic rank filter
                handler.addFilter(RankFilter())

    def get_logger(self):
        return self._logger

    def set_log_level(self, level: int):
        self._logger._logger.setLevel(level)
        for handler in self._logger._logger.handlers:
            handler.setLevel(level)
        self._logger.info(f"Logging level changed to: {logging.getLevelName(level)}")

    def set_log_all_ranks(self, enabled: bool):
        self.log_all_ranks = enabled
        self._update_filters()
        self._logger.info(f"log_all_ranks set to: {enabled}", all_ranks=True)


class LazyLogger:
    """Placeholder logger to avoid heavy imports until first actual log call."""

    def __init__(self):
        self._real_logger: DattaBotLoggerWrapper | None = None
        self._initializing = False  # Prevent recursion

    def _ensure_logger(self):
        global _logger_singleton
        if self._real_logger is None and not self._initializing:
            self._initializing = True
            try:
                # Initialize the actual singleton if needed
                if not isinstance(_logger_singleton, DattaBotLogger):
                    _logger_singleton = DattaBotLogger()
                self._real_logger = _logger_singleton._logger
            finally:
                self._initializing = False

    def __getattr__(self, name):
        # Avoid infinite recursion on special attributes
        if name in ("_real_logger", "_initializing"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        self._ensure_logger()
        return getattr(self._real_logger, name)


_logger_singleton: DattaBotLogger | LazyLogger = LazyLogger()


def get_logger(
    logging_level: int = None, log_all_ranks: bool = None
) -> DattaBotLoggerWrapper:
    """
    Return the singleton logger. If the logger already exists and a new
    logging level or log_all_ranks setting is passed in, it updates them.

    Args:
        logging_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_all_ranks: If True, log on all ranks. If False, only rank 0.
                      If None, keeps current setting (default: False)
    """
    global _logger_singleton

    # Initialize if still lazy
    if isinstance(_logger_singleton, LazyLogger):
        _logger_singleton = DattaBotLogger(
            logging_level=logging_level or logging.INFO,
            log_all_ranks=log_all_ranks if log_all_ranks is not None else False,
        )
        # Only log on rank 0
        if _get_current_rank() == 0:
            _logger_singleton._logger.info("Initialized logger!")
    else:
        # Update log level if needed
        if logging_level is not None:
            assert isinstance(
                logging_level, int
            ), f"Logging level must be an int, logging level passed in {logging_level}"
            current_level = _logger_singleton._logger._logger.level
            if logging_level != current_level:
                _logger_singleton.set_log_level(logging_level)
        # Update filter if needed (only if explicitly provided)
        if (
            log_all_ranks is not None
            and log_all_ranks != _logger_singleton.log_all_ranks
        ):
            _logger_singleton.set_log_all_ranks(log_all_ranks)

    return _logger_singleton._logger
