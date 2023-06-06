from loguru import logger

from src.gui import run


def main():
    logger.info('Hello!')
    return run()


if __name__ == '__main__':
    main()
