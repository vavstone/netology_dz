import gradio as gr
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def ner_pipeline_to_string(entities):
    """Преобразует выход NER пайплайна в единую строку"""
    if not entities:
        return ""
    grouped = {}
    for ent in entities:
        grp = ent['entity_group']
        word = ent['word'].strip()
        grouped.setdefault(grp, []).append(word)
    parts = []
    for grp, words in grouped.items():
        parts.append(f"{grp}: {', '.join(words)}")
    return "; ".join(parts)

def analyze(text: str) -> str:
    """Выполняет извлечение сущностей"""
    if not text.strip():
        return "Введите текст для анализа"

    ner = pipeline(
        "ner",
        model="Babelscape/wikineural-multilingual-ner",
        aggregation_strategy="simple",
        token=HF_TOKEN,
    )
    entities = ner(text)
    extracted_str = ner_pipeline_to_string(entities)
    return extracted_str

# Создаём интерфейс
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Текст для анализа", placeholder="Напишите отзыв...", lines=3),
    outputs=gr.Textbox(label="Результат"),
    title="Извлечение сущностей",
    description="Введите текст для извлечения сущностей",
    examples=[
        ["Стив Джобс основал Apple в Купертино"],
        ["Конференция пройдёт 15 мая 2025 года в Санкт-Петербурге"],
        ["Офис находится по адресу: г. Москва, ул. Тверская, д. 7."],
        ["Лионель Месси играет за футбольный клуб Интер"],
        ["Доктор Иванова назначила Аспирин пациенту Петрову"],
    ]
)

#analyze("Стив Джобс основал Apple в Купертино")
demo.launch()  # Откроется http://localhost:7860
# demo.launch(share=True)  # + публичная ссылка (на 72 часа)