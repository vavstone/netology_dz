import ollama
#from io import open

class ModelEfficiency:
    gen_tokens:int
    gen_duration:float
    gen_speed:float

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
        stream=True
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

#1. тест генерации контента по запросу для трех моделей
def generate_text():
    sysprompt = 'Ты - специалист по истории России. Отвечай на русском языке, вежливо и по существу.'
    prompt = 'Расскажи кратко об истории Москвы в 10 предложениях. Предложения нумеруй и начинай каждое с новой строки.'
    models = {'qwen3:1.7b','qwen3:8b','gemma3:4b'}
    print(f'#Тестирование генерации контента по запросу.')
    print(f'#Системный промпт: {sysprompt}')
    print(f'#Пользовательский промпт: {prompt}')
    print()
    print()
    for model in models:
        generate_text_for_model(model, sysprompt, prompt)

#2. тест извлечения сущностей из текста для трех моделей
def get_struct_from_text():
    in_data = '''1. ФИО: Смирнов Андрей Викторович
   Email: a.smirnov@mail.ru
   Телефон: +7-916-123-45-67
   Сумма долга: 12450 рублей
   Дата возникновения: 12.01.2026
   Статус: просрочка 58 дней
   Комментарий: клиент обещал оплатить до 20.03

2. ФИО: Козлова Елена Дмитриевна
   Email: elena.kozllova@yandex.ru
   Телефон: 8-495-789-12-34
   Сумма долга: 3500.50 рублей
   Дата возникновения: 01.03.2026
   Статус: оплачен
   Комментарий: чек №45678 от 05.04.2026

3. ФИО: Теймуразов Георгий Константинович
   Email: george.t@ge.com
   Телефон: +79265557788
   Сумма долга: 32000 рублей
   Дата возникновения: 05.02.2026
   Статус: просрочка 45 дней
   Комментарий: последнее уведомление отправлено 10.04

4. ФИО: Васильева Мария Петровна
   Email: m.vasilyeva2025@gmail.com
   Телефон: 89261234567
   Сумма долга: 0 рублей
   Дата возникновения: 20.02.2026
   Статус: оплачен
   Комментарий: квитанция №98765

5. ФИО: Park Sung-ho
   Email: sungpark@company.kr
   Телефон: +7-903-111-22-33
   Сумма долга: 18750 рублей
   Дата возникновения: 10.01.2026
   Статус: просрочка 72 дня
   Комментарий: требуется эскалация

6. ФИО: Иванова Анна Сергеевна
   Email: anneta_ivanova@bk.ru
   Телефон: 84951234567
   Сумма долга: 890.99 рублей
   Дата возникновения: 01.04.2026
   Статус: просрочка 14 дней
   Комментарий: клиент ожидает зарплату

7. ФИО: Новиков Павел Андреевич
   Email: p.novikov@inbox.ru
   Телефон: +7-906-777-88-99
   Сумма долга: 5200 рублей
   Дата возникновения: 25.03.2026
   Статус: оплачен частично
   Комментарий: остаток 5200 из 15000, новый срок 15.05'''
    #with open(file='./data/struct_data.txt', mode='r', encoding='UTF-8') as f:
    #    in_data = f.read()
    sysprompt = 'Ты — точный экстрактор данных. Отвечай только валидным JSON-массивом. Без пояснений.'
    prompt = f'''Из текста ниже извлеки ТОЛЬКО клиентов со статусом 'просрочка' и верни массив объектов:
[{{"name": "...", "email": "...", "debt": число, "days_overdue": число из статуса}}]

Текст:
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

#тест рассуждения
def reasoning():
    sysprompt = "Ты — логический анализатор. Рассуждай пошагово. Каждый шаг начинай с 'Шаг X:'. В конце дай краткий итог."
    prompt = """
Условие: У компании было 150 сотрудников. В первом квартале уволилось 20% сотрудников, но наняли на 15% больше, чем уволилось.
Во втором квартале уволилось 10% от текущего числа, а наняли столько же, сколько уволилось в первом квартале.
В третьем квартале уволилось 5 человек, а наняли 12.

Вопрос: Сколько сотрудников стало в конце третьего квартала?

Требование: Покажи каждый арифметический шаг отдельно. В конце напиши 'ОТВЕТ: число'.
"""
    models = {'qwen3:1.7b','qwen3:8b','gemma3:4b'}
    print(f'#Тестирование рассуждения.')
    print(f'#Системный промпт: {sysprompt}')
    print(f'#Пользовательский промпт: {prompt}')
    print()
    print()
    for model in models:
        generate_text_for_model(model, sysprompt, prompt)

#generate_text()
get_struct_from_text()
#reasoning()