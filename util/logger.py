from logging.handlers import RotatingFileHandler

import logging


def setup_logger(name: str, logging_level) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    sh = logging.StreamHandler()
    sh.setLevel(logging_level)
    log_format = '%(asctime)s -|- %(levelname)s -|- %(name)s -|- %(funcname)s-%(lineno)d -|- %(module)s -|- %(message)s'
    sh.setFormatter(logging.Formatter(log_format))
    return logger


def setup_logging(log_file_name: str, log_dir: str = "", logging_level: int = logging.INFO, log_to_file: bool = False):
    log_line_format = "%(asctime)s [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"

    main_logger = logging.getLogger()
    assert logging_level % 10 == 0 and logging_level / 10 <= 5
    main_logger.setLevel(logging_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(logging.Formatter(log_line_format))

    main_logger.addHandler(console_handler)
    if log_to_file:
        exp_file_handler = RotatingFileHandler(f'{log_dir}/{log_file_name}.log', maxBytes=10 ** 6, backupCount=5)
        exp_file_handler.setLevel(logging.DEBUG)
        exp_file_handler.setFormatter(logging.Formatter(log_line_format))
        main_logger.addHandler(exp_file_handler)
