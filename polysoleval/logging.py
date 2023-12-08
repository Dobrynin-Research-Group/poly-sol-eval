import logging
import os


def get_logger() -> logging.Logger:
    return logging.getLogger("uvicorn.error")


def setup_logger(log: logging.Logger) -> logging.Logger:
    levelnames = dict(
        debug=logging.DEBUG,
        info=logging.INFO,
        warning=logging.WARN,
        error=logging.ERROR,
        critical=logging.CRITICAL,
    )
    level = levelnames[os.environ.get("levelname", "info")]
    log.setLevel(level)

    # handler = logging.StreamHandler()
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    # formatter.default_time_format = "%H:%M:%S"
    # handler.formatter = formatter
    # log.addHandler(handler)
    return log
