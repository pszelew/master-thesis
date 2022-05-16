import logging
import sys
import os

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


def get_logger(name):
    logger = logging.getLogger(name)
    log_file_handler = logging.FileHandler(filename=os.path.join("logs", f"{name}.log"))
    logger.addHandler(log_file_handler)
    return logger
