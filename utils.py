import logging
from logging.handlers import TimedRotatingFileHandler

from config import Config


def setup_logger_handler(logger):
    console = logging.StreamHandler()
    path = Config.get().working_dir.joinpath("logs")
    path.mkdir(parents=True, exist_ok=True)
    file_path = path.joinpath('web-b-gone.log')
    file_handler = TimedRotatingFileHandler(
        filename=str(file_path),
        utc=True,
        when='midnight'
    )
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)-20s %(funcName)-16.16s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)


def get_threading_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if len(logger.handlers) < 1:
        console = logging.StreamHandler()
        path = Config.get().working_dir.joinpath("logs")
        path.mkdir(parents=True, exist_ok=True)
        file_path = path.joinpath('aramis_imarg_search_thread.log')
        file_handler = TimedRotatingFileHandler(
            filename=str(file_path),
            utc=True,
            when='midnight'
        )
        formatter = logging.Formatter(
            fmt='%(asctime)s %(threadName)-10s-%(thread)-6d %(name)-20s %(funcName)-16.16s %(levelname)-6s %(message)s',
            datefmt='%H:%M:%S'
        )
        console.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.debug('init done')
    return logger
