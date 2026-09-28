import logging
import sys

from argostranslate import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

if settings.debug:
    std_out_stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(std_out_stream_handler)


def debug(*argv):
    """Debug level log"""
    logger.debug(str(argv))


def info(*argv):
    """Info level log"""
    logger.info(str(argv))


def warning(*argv):
    """Warning level log"""
    logger.warning(str(argv))


def error(*argv):
    """Error level log"""
    logger.error(str(argv))