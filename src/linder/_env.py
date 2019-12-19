from logging.handlers import TimedRotatingFileHandler
import sys
import logging
import inspect
from pathlib import Path
import tempfile

########################################################################
# this file provides variable and functions useful for the whole module.
########################################################################

# define local path for loading resources in this package
path_module = Path(inspect.getsourcefile(lambda: 0)).resolve().parent

# set up logger format, note `u` to guarantee UTF-8 encoding
FORMATTER = logging.Formatter(u"%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# log file name
LOG_FILE = "linder.log"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    try:
        path_logfile = Path(LOG_FILE)
        path_logfile.touch()
    except Exception:
        tempdir = tempfile.gettempdir()
        path_logfile = Path(tempdir) / LOG_FILE

    file_handler = TimedRotatingFileHandler(
        path_logfile, when="midnight", encoding="utf-8",
    )
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    # better to have too much log than not enough
    logger.setLevel(level)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


logger_linder = get_logger("linder", logging.INFO)
logger_linder.debug("a debug message from linder")
