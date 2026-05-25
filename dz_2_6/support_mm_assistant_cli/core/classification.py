from models import Category

ESCALATION_KEYWORDS = {
    "позови",
    "переключи"
}

TECHNICAL_KEYWORDS = {
    "ошибка",
    "не работает",
    "не могу",
    "413",
    "500",
    "404",
    "загрузка",
    "вход",
    "синхронизация",
    "файл",
}

COMPLAINT_KEYWORDS = {
    "ужасно",
    "отстой",
    "недоволен",
    "плохо",
    "разочарован",
}

FAQ_KEYWORDS = {
    "спецификация",
    "функциональные требования",
    "описание БД"
}

def heuristic_classify(user_message: str) -> Category:
    text = user_message.lower()
    if any(keyword in text for keyword in ESCALATION_KEYWORDS):
        return Category.ESCALATION
    if any(keyword in text for keyword in COMPLAINT_KEYWORDS):
        return Category.COMPLAINT
    if any(keyword in text for keyword in TECHNICAL_KEYWORDS):
        return Category.TECHNICAL
    if any(keyword in text for keyword in FAQ_KEYWORDS):
        return Category.FAQ
    return Category.TECHNICAL

def should_escalate(user_message: str, category: Category, failed_attempts: int) -> bool:
    if category == Category.ESCALATION or failed_attempts > 3:
        return True
    text = user_message.lower()
    return any(keyword in text for keyword in ESCALATION_KEYWORDS)