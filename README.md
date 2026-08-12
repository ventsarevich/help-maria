# Поиск людей и объектов в видео с дрона

Локальный проект для macOS. Он работает без облачной загрузки видео и предоставляет два режима:

1. Статический анализ всех видео с формированием HTML-отчёта.
2. Просмотр отдельного видео с рамками YOLOE и горячими клавишами.

Статический анализ извлекает кадры через заданный интервал и рассматривает каждый кадр независимо. Движение дрона и соседние кадры не используются.

## Системные требования

- macOS;
- Apple Silicon M1/M2/M3/M4 рекомендуется;
- Python 3.11–3.13;
- свободное место для моделей и результатов;
- интернет при первом запуске для установки библиотек и загрузки моделей.

Проверьте Python:

```bash
python3 --version
```

Если команда отсутствует, установите Python с [python.org](https://www.python.org/downloads/macos/) или через Homebrew:

```bash
brew install python
```

## Получение и подготовка проекта

Склонируйте репозиторий и перейдите в него:

```bash
git clone https://github.com/ventsarevich/help-maria.git
cd help-maria
```

Если проект уже скачан, откройте Terminal, перейдите в его каталог и выполните:

```bash
cd /путь/к/help-maria
```

Создайте виртуальное окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Перед каждым новым сеансом Terminal активируйте окружение:

```bash
cd /путь/к/help-maria
source .venv/bin/activate
```

Все последующие команды в инструкции выполняются из корня проекта.

## Подготовка видео

Создайте папку `videos`, если её ещё нет:

```bash
mkdir -p videos
```

Положите в неё файлы `.MP4`, `.MOV` или `.M4V`. Подпапки также поддерживаются:

```text
help-maria/
├── videos/
│   ├── DJI_0001.MP4
│   ├── DJI_0002.MP4
│   └── another-flight/
│       └── DJI_0003.MOV
├── batch_analyze.py
└── video_viewer.py
```

Видео, модели, отчёты и извлечённые изображения исключены из Git через `.gitignore`.

## Быстрый старт: статический анализ

Интерактивный запуск:

```bash
export MPLCONFIGDIR=.cache/matplotlib
export PYTORCH_ENABLE_MPS_FALLBACK=1
python batch_analyze.py
```

Программа последовательно предложит выбрать:

1. Конфигурацию объектов.
2. Обычный или тройной анализ.
3. Модель для обычного анализа.
4. Глубину проверки кадров.

Для первого теста рекомендуется:

```text
full_search → Обычный → YOLOE → quick
```

После проверки скорости можно перейти на `balanced` или `deep`.

## Повторяемый запуск без меню

Быстрый YOLOE-анализ всех видео:

```bash
python batch_analyze.py \
  --analysis-mode single \
  --engine yoloe \
  --config full_search \
  --profile quick
```

Рекомендуемый сбалансированный анализ плитками:

```bash
python batch_analyze.py \
  --analysis-mode single \
  --engine yoloe \
  --config full_search \
  --profile balanced \
  --conf 0.12
```

Grounding DINO вместо YOLOE:

```bash
python batch_analyze.py \
  --analysis-mode single \
  --engine grounding-dino \
  --config full_search \
  --profile balanced \
  --conf 0.12
```

Тройной анализ YOLOE + Grounding DINO + SigLIP:

```bash
python batch_analyze.py \
  --analysis-mode triple \
  --config full_search \
  --profile quick
```

Тройной режим существенно медленнее и требует больше оперативной памяти. Сначала проверьте его на одном коротком видео.

## Профили анализа

| Профиль | Интервал | Обработка | Назначение |
|---|---:|---|---|
| `quick` | 5 секунд | полный кадр, 640 px | проверка установки и быстрый первый проход |
| `balanced` | 3 секунды | плитки 1280 px | основной рекомендуемый режим |
| `deep` | 1,5 секунды | плитки 960 px с большим перекрытием | медленный тщательный проход |

Плитки позволяют анализировать области исходного 4K-кадра без сильного уменьшения маленьких объектов.

## Пользовательские параметры

Указать собственные текстовые классы без изменения JSON:

```bash
python batch_analyze.py \
  --analysis-mode single \
  --engine yoloe \
  --classes "person,backpack,jacket" \
  --profile balanced
```

Проверять кадр каждые две секунды:

```bash
python batch_analyze.py \
  --analysis-mode single \
  --engine yoloe \
  --config people \
  --sample-seconds 2
```

Явно настроить плитки:

```bash
python batch_analyze.py \
  --analysis-mode single \
  --engine yoloe \
  --config full_search \
  --mode tiled \
  --tile-size 960 \
  --overlap 0.20 \
  --imgsz 960 \
  --conf 0.10
```

Использовать CPU вместо Apple GPU:

```bash
python batch_analyze.py \
  --device cpu \
  --analysis-mode single \
  --engine yoloe \
  --config full_search \
  --profile quick
```

Полный список аргументов:

```bash
python batch_analyze.py --help
```

## Конфигурации поиска

Готовые наборы находятся в `configurations.json`:

```json
{
  "person_only": ["person"],
  "people": ["person", "human", "man", "human body"],
  "clothing": ["jacket", "clothes"],
  "equipment": ["backpack"],
  "full_search": ["person", "human", "man", "backpack", "jacket", "clothes", "human body"]
}
```

Можно добавить собственный набор, сохранив корректный JSON.

## Веса итогового confidence

В тройном режиме коэффициенты моделей задаются в `analyzer_weights.json`:

```json
{
  "yoloe": 1.0,
  "grounding-dino": 0.9,
  "siglip": 0.65,
  "agreement_bonus": 0.12
}
```

`agreement_bonus` добавляется, когда один участок поддержали несколько анализаторов. Для каждого кандидата исходные оценки моделей сохраняются в `detections.json` и `detections.csv`.

## Результаты и HTML-отчёт

Каждый запуск создаёт отдельный каталог:

```text
results/YYYYMMDD-HHMMSS/
├── frames/          # полные кадры с рамкой
├── crops/           # подозрительные фрагменты
├── detections.json  # подробные данные
├── detections.csv   # таблица результатов
├── settings.json    # точная конфигурация запуска
└── report.html      # автономный отчёт
```

Открыть последний отчёт из Terminal:

```bash
open "$(find results -mindepth 2 -maxdepth 2 -name report.html | sort | tail -1)"
```

Отчёт работает без сервера. В нём доступны:

- группировка по видео;
- сортировка по confidence, согласию моделей или таймкоду;
- фильтр по названию видео и объекту;
- preview фрагмента и полный кадр;
- увеличение колесом или клавишами `+`/`-`;
- перетаскивание изображения;
- `←`/`→` для соседних кандидатов;
- `Esc` для возврата в каталог.

## Просмотр отдельного видео с AI-рамками

Интерактивный выбор видео и конфигурации:

```bash
export MPLCONFIGDIR=.cache/matplotlib
python video_viewer.py
```

После выбора объектов плеер предложит профиль качества:

| Профиль | Размер модели | Частота AI | Назначение |
|---|---:|---:|---|
| `fast` | 512 px | каждые 1,2 секунды видео | минимальная нагрузка, наиболее плавный просмотр |
| `balanced` | 640 px | каждые 0,65 секунды | рекомендуемый баланс |
| `quality` | 960 px | каждую 1 секунду | лучше для мелких объектов, но заметно тяжелее |

Профиль задаёт базовые значения. `--imgsz` и `--analysis-interval` позволяют переопределить их вручную.

Открыть конкретное видео с готовой конфигурацией:

```bash
python video_viewer.py \
  videos/DJI_0001.MP4 \
  --config full_search \
  --profile balanced \
  --device mps \
  --conf 0.12
```

Уменьшить нагрузку на Mac:

```bash
python video_viewer.py \
  videos/DJI_0001.MP4 \
  --config people \
  --profile fast \
  --imgsz 512 \
  --analysis-interval 1.2
```

### Горячие клавиши

| Клавиша | Действие |
|---|---|
| `Пробел` | пауза/продолжить |
| `←` / `A` | назад на 5 секунд |
| `→` / `D` | вперёд на 5 секунд |
| `J` | назад на 30 секунд |
| `L` | вперёд на 30 секунд |
| `1`, `2`, `4` | скорость воспроизведения |
| Колесо мыши или `+` / `-` | цифровой зум до 12× |
| Перетаскивание мышью | перемещение увеличенной области |
| `0` | сбросить зум и центр |
| `[` / `]` | понизить/повысить порог confidence |
| `S` | сохранить исходный кадр в `captures/` |
| `Q` / `Esc` | закрыть плеер |

Полный список параметров:

```bash
python video_viewer.py --help
```

## Запуск встроенных shell-скриптов

Скрипты `.command` выполняют создание окружения при необходимости, установку зависимостей и запуск программы.

Из Terminal:

```bash
./static_analyze.command
```

```bash
./video_viewer.command
```

Если macOS сообщает `Permission denied`:

```bash
chmod +x static_analyze.command video_viewer.command
```

Если Finder блокирует скачанный скрипт, запускайте его из Terminal командами выше.

## Остановка и повторный запуск

Остановить анализ в Terminal:

```text
Ctrl + C
```

Каждый повторный запуск создаёт новый каталог в `results`; предыдущие результаты не перезаписываются.

## Частые проблемы

### `No module named ...`

Активируйте окружение и повторно установите зависимости:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Ошибка MPS

Временно запустите на CPU:

```bash
python batch_analyze.py --device cpu --analysis-mode single --engine yoloe --config full_search --profile quick
```

### Недостаточно памяти

- используйте `single`, а не `triple`;
- выберите `quick`;
- увеличьте `--sample-seconds`;
- для плеера уменьшите `--imgsz` и увеличьте `--analysis-interval`;
- закройте приложения, активно использующие память и GPU.

### Модель загружается при первом запуске

Это нормально. YOLOE, Grounding DINO и SigLIP автоматически скачиваются в кеш библиотек и не должны добавляться в Git. Следующие запуски используют локальные копии.

### Слишком много ложных кандидатов

Поднимите `--conf`, например:

```bash
python batch_analyze.py --analysis-mode single --engine yoloe --config full_search --profile balanced --conf 0.20
```

### Модель пропускает маленькие объекты

Используйте `balanced` или `deep`, уменьшите `--tile-size`, увеличьте `--imgsz` и при необходимости понизьте `--conf`. Это замедляет обработку и увеличивает число ложных находок.

## Ограничения

- Низкий confidence не доказывает наличие объекта.
- Универсальные модели могут принимать камни, тени и растительность за человека или одежду.
- Если объект занимает несколько пикселей, достоверно восстановить отсутствующие детали невозможно.
- Любой результат требует ручной проверки исходного кадра и соседних моментов видео.
