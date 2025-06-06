import logging


class UnixTimeFormatter(logging.Formatter):

  def formatTime(self, record, datefmt=None):
    return str(int(record.created))


def get_logger(name="AppLogger"):
  logger = logging.getLogger(name)

  if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    formatter = UnixTimeFormatter('[%(asctime)s] [%(levelname)s] %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

  return logger
