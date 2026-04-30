import ollama
from pydantic import BaseModel
from typing import List

class ModelEfficiency:
    gen_tokens:int
    gen_duration:float
    gen_speed:float

class Entities(BaseModel):
    persons:List[str]
    organizations:List[str]
    dates:List[str]


#генерация контента по запросу
def generate_text_for_model(model: str, sysprompt: str, prompt:str) -> ModelEfficiency:
    print(f'#Модель: {model}.')
    print(f'#Сгенерированный контент:')
    client = ollama.Client(host='http://localhost:11434')
    stream = client.chat(
        model=model,
        messages=[
            {"role":"system", "content":sysprompt},
            {"role":"user", "content":prompt}
        ],
        stream=True,
        format=Entities.model_json_schema()
    )
    res = ModelEfficiency()
    for chunk in stream:
        content = chunk["message"]["content"]
        if content:
            print(content,end='',flush=True)
        eval_count = chunk.get("eval_count")
        if eval_count:
            res.gen_tokens = eval_count
        eval_duration = chunk.get("eval_duration")
        if eval_duration:
            res.gen_duration = eval_duration / 1e9
    print()
    if res.gen_duration>0 and res.gen_tokens>0:
        res.gen_speed = res.gen_tokens/res.gen_duration
    print()
    print(f'#Показатели: токенов - {res.gen_tokens}, время генерации - {res.gen_duration:.2f} сек, скорость - {res.gen_speed:.2f} токенов/сек')
    print()
    print()
    return res



#тест извлечения сущностей из текста для трех моделей
def get_struct_from_text():
    in_data = '''В период с 15 января по 20 февраля 2026 года компания ООО «ТехноИнновации» провела серию встреч с представителями ПАО «Газпром». Со стороны «Газпрома» в переговорах участвовали Алексей Борисович Кузнецов (заместитель председателя правления) и Ирина Владимировна Соколова (начальник департамента). 25 января 2026 года состоялось подписание меморандума между ООО «ТехноИнновации» и АО «РосАтом». Документ подписали генеральный директор «ТехноИнноваций» Михаил Петрович Зайцев и советник руководителя «РосАтома» Елена Павловна Ковальчук.
15 марта 2026 года на конференции «Цифровая экономика – 2026», организованной АО «РосБизнесСофт» при поддержке ГК «ТехноЛидер», выступили: Дмитрий Николаевич Власов (CTO, АО «РосБизнесСофт»), Стивен Кинг (FutureLabs Inc., США) и Мария Ивановна Петрова (независимый эксперт). Конференция проходила в Москве с 15 по 17 марта 2026 года.
10 апреля 2026 года состоялось внеочередное собрание акционеров ПАО «Северсталь». На собрании присутствовали председатель совета директоров Алексей Александрович Мордашов и представитель ООО «Инвестиционная палата» Сергей Павлович Романов.
Также в апреле 2026 года, а именно 20.04.2026, стартап "Квантовый скачок", основанный физиком Дмитрием Алексеевичем Вороновым в 2020 году, объявил о привлечении инвестиций от венчурного фонда «ТехноВенчурс».'''
    sysprompt = 'Ты — система извлечения сущностей. Извлеки из текста все уникальные имена людей, организации и даты. Ответь строго JSON с полями persons, organizations, dates.'
    prompt = f'''Текст:
{in_data}
'''
    models = {'qwen3:1.7b','qwen3:8b','gemma3:4b'}
    print(f'#Тестирование извлечения сущностей из текста')
    print(f'#Системный промпт: {sysprompt}')
    print(f'#Пользовательский промпт: {prompt}')
    print()
    print()
    for model in models:
        generate_text_for_model(model, sysprompt, prompt)


get_struct_from_text()