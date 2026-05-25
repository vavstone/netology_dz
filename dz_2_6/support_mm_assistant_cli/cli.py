from config import Settings
from loguru import logger
from core.assistant import SupportAssistantApp
from utils import get_image_info
from models import AssistantResponse, AssistantAppState, ImageInfo


def setup_logging(settings: Settings) -> None:
    """Настраивает логирование единожды."""
    logger.remove()  # удаляем стандартный хендлер (вывод в консоль)
    logger.add(settings.log_path, format="{time} {message}", rotation="10 MB")
    # можно добавить и вывод в консоль, если нужно:
    # logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

def main():
    settings = Settings.from_env()
    setup_logging(settings)
    assistant = SupportAssistantApp(settings)

    print(f"=== {settings.service_name} Support CLI ===")
    print("Команды: /main_menu, /clear, /stats, /send_img, /quit")

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            return None

        if not user_input:
            continue

        # Обработка команд
        if user_input.startswith("/"):
            command_result = assistant.handle_command(user_input)
            if command_result is None:
                print("До свидания!")
                return None
            print(command_result)
            continue

        #Ожидаем от пользователя отправки пути к изображению
        if assistant.state == AssistantAppState.WAITING_FOR_IMG:
            try:
                image_info = get_image_info(user_input)
                #если не изображение, не идем дальше
                if not image_info.is_valid_image():
                    print("Можно использовать только изображения!")
                    continue
                assistant.image_info = image_info
                assistant.state = AssistantAppState.WAITING_FOR_IMG_QUESTION
                print("Задайте вопрос, связанный с этим изображением:")
                continue
            except Exception as e:
                print(f"Ошибка чтения файла {e}")
                print(f"Убедитесь, что файл {user_input} существует и повторите попыту")
                continue

        image_info = None
        # Ожидаем от пользователя отправки вопроса к изображению
        if assistant.state == AssistantAppState.WAITING_FOR_IMG_QUESTION:
            image_info = assistant.image_info

        # Потоковый вывод ответа
        try:
            response_gen = assistant.respond(user_input, image_info)
        except Exception as e:
            print(f"Ошибка при инициализации: {e}")
            continue

        print()  # перевод строки перед ответом
        final_response: AssistantResponse
        try:
            while True:
                try:
                    chunk = next(response_gen)
                    # Печатаем порцию текста без новой строки и сразу сбрасываем буфер
                    print(chunk, end='', flush=True)
                except StopIteration as e:
                    final_response = e.value   # AssistantResponse
                    break
        except Exception as e:
            print(f"\nОшибка во время генерации ответа: {e}")
            continue

        # После завершения потока показываем метаданные
        if final_response is not None:
            source = "cache" if final_response.from_cache else final_response.provider
            print(f"\n[{final_response.category} | {source} | {final_response.latency_seconds:.2f}с]")
        else:
            print("\n[Ошибка: ответ не получен]")