import sys
from pathlib import Path

# Определяем папку приложения (там, где находится этот скрипт)
APP_DIR = Path(__file__).parent
LOG_FILE = APP_DIR / "output.log"  # имя файла для сохранения вывода

def tee_print(*args, sep=' ', end='\n', file=sys.stdout, flush=False):
    """
    Аналог print, который:
      - выводит в консоль (или в указанный file)
      - дописывает такую же строку в текстовый файл (LOG_FILE)
    """
    # 1. Вывод в обычный print (консоль или другой поток)
    print(*args, sep=sep, end=end, file=file, flush=flush)

    # 2. Формируем ту же строку для записи в файл
    line = sep.join(str(arg) for arg in args) + end

    # 3. Пишем в файл (в режиме добавления, UTF-8)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line)
        if flush:
            f.flush()
