from loguru import logger

logger.add("log/log.log", level="INFO", rotation="100 MB", retention="3 year")
