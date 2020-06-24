import importlib
import os


def get_logger(module, logdir=None):
    logger = importlib.import_module(f"tfmpc.loggers.{module}")

    if logdir:
        filename = os.path.join(logdir, "trace.log")
        logger.setup(filename=filename)

    return logger
