import base64
import os

from models import ImageInfo


def get_image_info(image_path: str) -> ImageInfo:
    """
    Читает изображение из файла и возвращает объект ImageInfo,
    содержащий имя файла, полный путь, размер, base64-строку и расширение.
    """
    # Размер файла в байтах
    size = os.path.getsize(image_path)

    # Имя файла с расширением (например, "cat.jpg")
    file_name = os.path.basename(image_path)

    # Полный путь к файлу (абсолютный или как передан)
    file_full_name = os.path.abspath(image_path)

    # Расширение без точки (например, "jpg")
    extension = os.path.splitext(image_path)[1].lower().lstrip('.')

    # Чтение и кодирование содержимого в Base64
    with open(image_path, "rb") as image_file:
        base64content = base64.b64encode(image_file.read()).decode("utf-8")

    # Создание и возврат объекта ImageInfo
    return ImageInfo(
        file_name=file_name,
        file_full_name=file_full_name,
        size=size,
        base64content=base64content,
        extension=extension
    )