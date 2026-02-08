import logging
from src.ui.printer import printer
from src.config.settings import LOG, LOG_LEVEL, LOG_TO_FILE, LOG_TO_UI


class Logger:
    _logger = None
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    _logging_enabled = LOG
    _use_fancy_print = LOG_TO_UI
    _use_file_handler = LOG_TO_FILE

    @classmethod
    def get_logger(
        cls,
        name: str = "deepshell",
        level: str = LOG_LEVEL,
        log_file: str = "deepshell.log",
    ) -> logging.Logger:
        """
        Returns a logger instance with a selectable log level, logging only to a file.
        """
        if cls._logger is None:
            cls._logger = logging.getLogger(name)
            log_level = cls.LEVELS.get(level.lower(), logging.INFO)
            cls._logger.setLevel(log_level)

            if cls._use_file_handler and cls._logging_enabled:
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                cls._logger.addHandler(file_handler)

            if cls._use_fancy_print and cls._logging_enabled:
                cls._logger.addHandler(FancyPrintHandler())

        return cls._logger


class FancyPrintHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit the log record using fancy_print with colors.
        """
        try:
            msg = self.format(record)
            colored_msg = self._apply_color(msg, record.levelno)
            level_name = record.levelname
            formatted_msg = f"[{level_name}] {msg}"
            colored_msg = self._apply_color(formatted_msg, record.levelno)

            printer(colored_msg)
        except Exception:
            self.handleError(record)

    def _apply_color(self, msg: str, level: int) -> str:
        """
        Apply color formatting based on the log level.
        """
        color_map = {
            logging.DEBUG: "blue",
            logging.INFO: "green",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
            logging.CRITICAL: "purple",
        }

        color = color_map.get(level, "white")

        return f"[{color}]{msg}[/]"
