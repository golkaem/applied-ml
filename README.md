# Прикладные задачи МО
Репозиторий с домашними заданиями.

## Домашнее задание №1

1. Найти и согласовать датасет.
2. Обучить как минимум 2 архитектуры сетей и получить бейзлайн.
3. Сравнить, как влияют на качество моделей техники:
   - итеративная аугментация;
   - итеративная разморозка слоев с адаптивным learning rate;
   - псевдолейблинг.

Решение находится в папке `hw1/`.

## Окружение

Проект рассчитан на Python 3.11 и CPU. Создание окружения:

```bash
python -m venv venv
```

Активация в Git Bash на Windows:

```bash
source venv/Scripts/activate
```

В `cmd.exe` используется команда `venv\Scripts\activate`.
В Linux/macOS — `source venv/bin/activate`.

Для CPU-сборки PyTorch на Windows:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

Остальные зависимости:

```bash
pip install -r requirements.txt
```
