class Settings:
    MAIN_WINDOW_TITLE: str = 'Цифровая обработка изображений с использованием преобразования Уолша-Адамара'
    MAIN_WINDOW_WIDTH: int = 768 * 2
    MAIN_WINDOW_HEIGHT: int = 768 + 100

    ENABLED_IMAGE_FILETYPES: list[str] = [
        "*.bmp", "*.tiff", "*.png",
        "*.jpg", "*.jpeg", "*.jfif",
    ]
    IMAGE_FRAME_SIZE = 512 + 32


settings = Settings
