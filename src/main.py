from loguru import logger


def main():
    logger.info('Hello!')


def foo() -> str:
    return "bar"
