from config import Settings
from core.assistant import SupportAssistantApp
from models import AssistantResponse


def main():
    settings = Settings.from_env()
    assistant = SupportAssistantApp(settings)

    print(f"=== {settings.service_name} Support CLI ===")
    print("Команды: /clear, /stats, /quit")

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            return None

        if not user_input:
            continue

        # Обработка команд (без изменений)
        if user_input.startswith("/"):
            command_result = assistant.handle_command(user_input)
            if command_result is None:
                print("До свидания!")
                return None
            print(command_result)
            continue

        # Потоковый вывод ответа
        try:
            response_gen = assistant.respond(user_input)
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