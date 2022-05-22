import logging
import sys
import os

import tqdm

if not os.path.exists("logs"):
    os.mkdir("logs")

file_handler = logging.FileHandler(filename=os.path.join("logs", "tmp.log"))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name):
    logger = logging.getLogger(name)
    log_file_handler = logging.FileHandler(filename=os.path.join("logs", f"{name}.log"))
    logger.addHandler(log_file_handler)
    logger.addHandler(TqdmLoggingHandler())
    return logger
