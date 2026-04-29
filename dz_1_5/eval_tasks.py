EVAL_TASKS = [
    {
        "name": "Простой текст: персона, организация, место",
        "prompt": "Стив Джобс основал Apple в Купертино.",
        "check": lambda r: all(e in r for e in ["Стив Джобс", "Apple", "Купертино"])
    },
    {
        "name": "Дата и место",
        "prompt": "Конференция пройдёт 15 мая 2025 года в Санкт-Петербурге.",
        "check": lambda r: "15 мая 2025" in r and "Санкт-Петербург" in r
    },
    {
        "name": "Персона и организация",
        "prompt": "Генеральный директор Microsoft Сатья Наделла выступил на конференции.",
        "check": lambda r: "Сатья Наделла" in r and "Microsoft" in r
    },
    {
        "name": "Геополитические сущности",
        "prompt": "Франция и Германия подписали договор в Брюсселе.",
        "check": lambda r: all(e in r for e in ["Франция", "Германия", "Брюссель"])
    },
    {
        "name": "Денежная сумма и фонд",
        "prompt": "Стартап привлёк инвестиции в размере 10 миллионов долларов от Sequoia Capital.",
        "check": lambda r: "10 миллионов долларов" in r and "Sequoia Capital" in r
    },
    {
        "name": "Процент, страна и дата",
        "prompt": "Инфляция в России составила 7,5% в марте 2024 года.",
        "check": lambda r: all(e in r for e in ["Россия", "7,5%", "март 2024"])
    },
    {
        "name": "Продукт и компания",
        "prompt": "Samsung выпустила новый смартфон Galaxy S24 Ultra.",
        "check": lambda r: "Samsung" in r and "Galaxy S24 Ultra" in r
    },
    {
        "name": "Адрес",
        "prompt": "Офис находится по адресу: г. Москва, ул. Тверская, д. 7.",
        "check": lambda r: "Москва" in r and "Тверская" in r
    },
    {
        "name": "Фильм и год",
        "prompt": "Фильм Интерстеллар получил Оскар в 2015 году.",
        "check": lambda r: "Интерстеллар" in r and "2015" in r
    },
    {
        "name": "Спортсмен и клуб",
        "prompt": "Лионель Месси играет за Интер Майами.",
        "check": lambda r: "Лионель Месси" in r and "Интер Майами" in r
    },
    {
        "name": "Врач, пациент, лекарство",
        "prompt": "Доктор Иванова назначила Аспирин пациенту Петрову.",
        "check": lambda r: all(e in r for e in ["Иванова", "Петров", "Аспирин"])
    },
    {
        "name": "Книга, автор, год",
        "prompt": "Роман Война и мир написал Лев Толстой в 1869 году.",
        "check": lambda r: all(e in r for e in ["Война и мир", "Лев Толстой", "1869"])
    },
    {
        "name": "Технические сущности",
        "prompt": "Сервер на базе Intel Xeon размещён в дата-центре в Нидерландах.",
        "check": lambda r: all(e in r for e in ["Intel", "Xeon", "Нидерланды"])
    },
    {
        "name": "Событие и место",
        "prompt": "Олимпийские игры 2024 прошли в Париже.",
        "check": lambda r: all(e in r for e in ["Олимпийские игры", "2024", "Париж"])
    },
    {
        "name": "Email и телефон",
        "prompt": "Свяжитесь с нами по email support@example.com или телефону +7 495 123-45-67.",
        "check": lambda r: "support@example.com" in r and "+7 495 123-45-67" in r
    },
    {
        "name": "Ник в соцсети",
        "prompt": "Подпишитесь на @elonmusk в Твиттере.",
        "check": lambda r: "elonmusk" in r.lower()   # модель может вернуть без @
    },
    {
        "name": "Номер рейса и города",
        "prompt": "Рейс SU1234 из Москвы в Нью-Йорк задержан.",
        "check": lambda r: "SU1234" in r and "Москва" in r and "Нью-Йорк" in r
    },
    {
        "name": "URL",
        "prompt": "Подробнее на сайте https://example.com/news.",
        "check": lambda r: "https://example.com/news" in r
    },
    {
        "name": "Количество и валюта",
        "prompt": "Доставка 5 товаров на сумму 1500 рублей.",
        "check": lambda r: "5" in r and "1500 рублей" in r
    },
    {
        "name": "Цитата и автор",
        "prompt": "Как сказал Альберт Эйнштейн: Воображение важнее знания.",
        "check": lambda r: "Альберт Эйнштейн" in r
    }
]