from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
from tee_print import tee_print

load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))

# Поиск моделей для NER, отсортированных по скачиваниям
tee_print("Топ-100 моделей для ner:\n")
models = api.list_models(
    pipeline_tag="token-classification",
    sort="downloads",
    limit=100
)

for model in models:
    # Получаем детальную информацию о конкретной модели
    tee_print("=" * 50)
    tee_print(f"  {model.id}")
    tee_print(f"    Скачиваний: {model.downloads:,}")
    tee_print(f"    Лайков: {model.likes:,}")
    tee_print(f"    Теги: {', '.join(model.tags[:5]) if model.tags else 'нет'}")
    info = api.model_info(model.id)
    tee_print(f"    ID: {info.id}")
    tee_print(f"    Автор: {info.author}")
    tee_print(f"    Скачиваний: {info.downloads:,}")
    tee_print(f"    Лицензия: {info.card_data.get('license', 'не указана') if info.card_data else 'нет данных'}")
    tee_print(f"    Теги: {', '.join(info.tags[:8]) if info.tags else 'нет'}")
    tee_print(f"    Размер: {info.safetensors.total if info.safetensors else 'неизвестно'} параметров")
    tee_print()


