# План тестирования YDB в VectorDBBench и анализ достаточности кода

Документ описывает план запуска бенчмарков векторного поиска для сравнения
**YDB** с популярными векторными СУБД (с акцентом на горизонтальное
масштабирование YDB) и проверяет, достаточно ли текущей реализации
(`vectordb_bench/backend/clients/ydb/`) для его выполнения.

---

## 1. Цели и методология

YDB — распределённая СУБД, и её ключевое отличие от большинства движков
лидерборда — **горизонтальное масштабирование**: данные разбиваются на таблетки
(DataShard), распределяемые по узлам, а SDK-драйвер сам балансирует запросы.
План строится вокруг двух осей:

1. **Сопоставимость** — YDB и конкуренты прогоняются на одинаковых датасетах,
   кейсах и нагрузке клиента (`--case-type`, `--k`, `--num-concurrency`,
   `--concurrency-duration`), чтобы сравнивать QPS, p99-latency и recall.
2. **Масштабируемость** — YDB прогоняется на кластерах разного размера
   (1 → 3 → 9 узлов) на одном кейсе, чтобы измерить рост QPS/throughput и
   поведение latency/recall при добавлении узлов.

Размер кластера — операционный параметр развёртывания; бенчмарк его не задаёт,
различия фиксируются через `--db-label` (туда удобно кодировать размер/тип
инстанса).

**Метрики на кейс:** `insert_duration`, `optimize_duration`, `load_duration`;
`recall`, `ndcg`, `serial_latency_p99/p95`; `qps` и по-конкуррентностям
`conc_qps_list`, `conc_latency_p99/p95/avg_list`.

**Предпосылки:** одинаковый клиентский хост во всех прогонах (желательно в той же
зоне, что и сервер); метрика расстояния берётся из датасета автоматически
(`assembler.py`); каждый кейс изолируется своей таблицей (`--table-name` или
авто-имя по case-type).

---

## 2. Подготовка окружения

### 2.1. Конкуренты

| Класс | СУБД | Команда CLI |
|------|------|-------------|
| PostgreSQL + HNSW | pgvector | `vectordbbench pgvectorhnsw` |
| PostgreSQL + IVF/RaBitQ | VectorChord | `vectordbbench vectorchordrq` |
| Специализированный ANN | Milvus | `vectordbbench milvushnsw` |
| Специализированный ANN | Qdrant | `vectordbbench qdrant` |
| Поисковый движок | Elasticsearch/OpenSearch | `vectordbbench elasticcloudhnsw` / `awsopensearch` |
| Распределённый HTAP + вектор | **YDB** | `vectordbbench ydb` |

Для сравнения «масштабируемость против масштабируемости» берём системы, которые
тоже шардируются (Milvus distributed, OpenSearch, Qdrant); single-node движки
(pgvector) — как однопроцессный baseline.

### 2.2. Кластеры YDB

Готовим минимум 3 конфигурации одного кластера, отличающиеся числом узлов:
`ydb-1n`, `ydb-3n`, `ydb-9n`. Подключение — через endpoint discovery:

```shell
export YDB_ENDPOINT=grpc://<lb-or-any-node>:2136
export YDB_DATABASE=/Root/db1
# при необходимости: YDB_USER / YDB_PASSWORD или SDK-креды
```

Параметры авто-партиционирования задаются на стороне подключения (одинаково для
всех прогонов кластера). Минимум партиций имеет смысл держать не ниже числа ядер
CPU базы; дефолт `--auto-partitioning-min-partitions-count 1000` рассчитан на
крупные кластеры, для 1–3 узлов его стоит уменьшить:

```shell
--auto-partitioning-min-partitions-count 50 \
--auto-partitioning-max-partitions-count 200 \
--auto-partitioning-partition-size-mb 1000
```

Партиции не предсоздаются (`UNIFORM_PARTITIONS` не используется), поэтому таблица
стартует с одного шарда и распределяется по узлам по мере роста нагрузки — отсюда
обязательная фаза прогрева (§4, фаза 2).

### 2.3. Датасеты

По умолчанию датасеты качаются в `/tmp/vectordb_bench/dataset`
(`DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}`). `/tmp` часто чистится при
перезагрузке, поэтому большие датасеты перекачиваются. Чтобы скачивать один раз,
задайте `DATASET_LOCAL_DIR` на постоянный быстрый диск — через окружение или
`.env` в каталоге запуска (один и тот же путь для CLI и UI):

```shell
export DATASET_LOCAL_DIR=/data/vectordb_bench/dataset
```

Источник можно сменить через `DATASET_SOURCE` (`S3` по умолчанию / `AliyunOSS`)
и `DEFAULT_DATASET_URL`. Файлы скачиваются повторно только при несовпадении
размера, поэтому локальную копию можно подготовить заранее (например
`aws s3 cp --no-sign-request s3://assets.zilliz.com/benchmark/<dataset>/<dir>/ \
$DATASET_LOCAL_DIR/<dataset>/<dir>/ --recursive`).

---

## 3. Матрица датасетов и кейсов

| Ось | Кейсы | Назначение |
|-----|-------|-----------|
| Размер (768d, Cohere) | `Performance768D1M` → `Performance768D10M` | масштаб по объёму |
| Размер (1536d, OpenAI) | `Performance1536D500K` → `Performance1536D5M` | большая размерность |
| Малый (smoke) | `Performance1536D50K` | быстрый прогон/отладка |
| Int-фильтр | `Performance768D1M1P` / `...99P`, `Performance1536D500K1P` / `...99P` | фильтр по диапазону id |
| Label-фильтр | `LabelFilterPerformanceCase` (+ `--dataset-with-size-type`) | фильтр по равенству метки |

Свип для кривой recall/QPS: `--kmeans-tree-search-top-size` ∈ {1,4,10,16,32},
`--overlap-clusters` ∈ {1,3,5}, опционально `--levels`/`--clusters`.

---

## 4. Порядок запуска: три фазы

Все прогоны — при активном окружении (`pip install -e '.[ydb]'`) и выставленных
`YDB_ENDPOINT`/`YDB_DATABASE`. `--num-concurrency`/`--concurrency-duration`
держим одинаковыми для YDB и конкурентов.

Идея разбиения: данные грузятся и индексируются **один раз на кластер** (фаза 1),
затем кластер прогревается (фаза 2), и только потом снимаются зачётные метрики
(фаза 3, в которой данные не перезагружаются — `--skip-drop-old --skip-load`).

### Фаза 1 — Загрузка данных и построение индекса

Выполняет drop-old + load + построение `vector_kmeans_tree` (этап `optimize`).
Запускается один раз на кластер для каждого кейса.

Сначала smoke-прогон на малом датасете для проверки конвейера end-to-end:

```shell
vectordbbench ydb --case-type Performance1536D50K --db-label ydb-1n-smoke \
  --drop-old --load --skip-search-serial --skip-search-concurrent \
  --auto-partitioning-min-partitions-count 8 --auto-partitioning-max-partitions-count 64
```

Затем загрузка зачётного датасета (пример — Cohere 10M):

```shell
vectordbbench ydb --case-type Performance768D10M --db-label ydb-1n \
  --drop-old --load --skip-search-serial --skip-search-concurrent \
  --kmeans-tree-search-top-size 10
```

Здесь же снимается `insert_duration`/`optimize_duration`. Учитывайте, что
монотонный `Uint64`-ключ создаёт «горячий» хвостовой шард на старте загрузки,
поэтому прирост throughput записи от добавления узлов может быть занижен
(см. §5, H). На больших датасетах (5M/10M) построение индекса может длиться
дольше захардкоженного таймаута ожидания (см. §5, D и §6).

### Фаза 2 — Прогрев

Так как партиции таблицы и индекса разъезжаются по узлам по мере роста нагрузки,
«холодный» замер сразу после загрузки занижает масштабирование. Перед зачётными
тестами дайте поисковую нагрузку на прогрев и дождитесь стабилизации числа
партиций и их распределения по узлам (видно в мониторинге YDB):

```shell
vectordbbench ydb --case-type Performance768D10M --db-label ydb-1n-warmup \
  --skip-drop-old --skip-load --skip-search-serial --search-concurrent \
  --num-concurrency 80 --concurrency-duration 300 \
  --kmeans-tree-search-top-size 10
```

Результаты прогрева не учитываются; цель — раздать шарды по узлам.

### Фаза 3 — Выполнение тестов

Снимаем зачётные метрики на уже загруженных и прогретых данных
(`--skip-drop-old --skip-load`). Ниже — варианты тестов.

**3.1. Baseline (1 узел), medium-датасеты:**

```shell
vectordbbench ydb --case-type Performance768D1M --db-label ydb-1n \
  --skip-drop-old --skip-load --search-serial --search-concurrent \
  --num-concurrency 1,5,10,20,30,40,60,80 --concurrency-duration 30 \
  --kmeans-tree-search-top-size 10
```

**3.2. Горизонтальное масштабирование** — один кейс на 1/3/9 узлах, меняется
только кластер и `--db-label` (для каждого кластера предварительно выполнены
фазы 1–2):

```shell
vectordbbench ydb --case-type Performance768D10M --db-label ydb-3n \
  --skip-drop-old --skip-load --search-serial --search-concurrent \
  --num-concurrency 1,10,20,40,60,80,100,120 --concurrency-duration 60 \
  --kmeans-tree-search-top-size 10
```

Список конкуррентностей доводим до выхода QPS на плато — «потолок» при разном
числе узлов и показывает масштабируемость.

**3.3. Масштабирование по объёму** — фиксируем кластер, меняем датасет
(`Performance768D1M` → `Performance768D10M`), повторяя фазы 1–3 для каждого.

**3.4. Фильтрация:**

```shell
# label-фильтр (равенство метки)
vectordbbench ydb --case-type LabelFilterPerformanceCase \
  --dataset-with-size-type "Medium Cohere (768dim, 1M)" --db-label ydb-3n \
  --drop-old --load --search-serial --search-concurrent
```

Label-фильтр работает штатно (индекс `ON (labels, embedding)`, поиск по равенству
метки). Int-фильтр (`*1P/*99P`) перед публикацией требует проверки recall и
статистики запроса — префикс по уникальному `id` плохо ложится на фильтрованный
индекс (см. §5, C).

**3.5. Кривая recall ↔ QPS** — свип `--kmeans-tree-search-top-size` без
перезагрузки данных; `--overlap-clusters` влияет на построение, поэтому его свип
требует отдельной фазы 1 (`--drop-old --load`):

```shell
for TOP in 1 4 10 16 32; do
  vectordbbench ydb --case-type Performance768D1M --db-label "ydb-3n-top$TOP" \
    --skip-drop-old --skip-load --search-serial --search-concurrent \
    --num-concurrency 1,10,50,80 --concurrency-duration 30 \
    --kmeans-tree-search-top-size $TOP
done
```

**3.6. Конкуренты** — те же кейсы/`--k`/`--num-concurrency`/`--concurrency-duration`.
Пример pgvector:

```shell
vectordbbench pgvectorhnsw --case-type Performance768D1M --db-label pg-hnsw \
  --user-name ... --password ... --host ... --db-name ... \
  --m 16 --ef-construction 128 --ef-search 128 \
  --num-concurrency 1,5,10,20,30,40,60,80 --concurrency-duration 30
```

Все системы удобно собрать в batch-конфиг и запустить одной командой
`vectordbbench batchcli --batch-config-file plan.yaml`. Итоговое сравнение —
в UI (`init_bench`), где результаты YDB и эталонные прогоны видны рядом.

---

## 5. Достаточность кода и недоработки

Что уже поддержано: регистрация клиента в CLI/UI (иконка, параметры кейса);
полный цикл (`init`/`insert`/`optimize`/`search`/`prepare_filter`);
авто-партиционирование таблицы и внутренних таблиц индекса
(`AUTO_PARTITIONING_BY_SIZE/BY_LOAD`, настраиваемые `MIN/MAX_PARTITIONS_COUNT`,
`PARTITION_SIZE_MB`); авто-выбор метрики из датасета; масштабирование
конкурентного поиска (свой драйвер/пул на процесс); изоляция кейсов по таблицам;
параметры индекса/поиска в CLI и UI; корректный label-фильтр; авто-подбор формы
дерева; ожидание готовности индекса; тесты (`tests/test_ydb.py`).

Для сопоставимого сравнения без фильтров и для масштабирования (при соблюдении
прогрева) кода достаточно. Открытые недоработки (по приоритету):

- **D.** Таймаут ожидания индекса захардкожен (`YDB_INDEX_WAIT_TIMEOUT_SECONDS =
  7200`) и не связан с `optimize_timeout`: на 5M/10M построение может длиться
  дольше 2 ч и упасть раньше времени. → пробросить `optimize_timeout` или сделать
  таймаут опцией.
- **C.** Int-фильтр строит индекс с префиксом по уникальному `id` и фильтрует
  диапазоном `id >= value`, что плохо ложится на фильтрованный kmeans-tree (риск
  низкого recall / full-scan). → проверить план; при подтверждении хранить
  отдельную фильтр-колонку или пометить кейсы экспериментальными. Label-фильтр
  проблемы лишён.
- **G′.** Нет начального предразбиения. `MIN_PARTITIONS_COUNT` не предсоздаёт
  партиции; `UNIFORM_PARTITIONS` для монотонных id неэффективен. → опция
  `PARTITION_AT_KEYS` по ожидаемому размеру датасета (убирает требование прогрева,
  ускоряет построение — см. §6).
- **H.** Монотонный `Uint64` PK → горячий шард на загрузке. → предразбиение /
  «размазывание» ключа; минимум — оговорка при чтении `insert_duration`.
- **E.** Размер батча вставки захардкожен (1000), `NUM_PER_BATCH` игнорируется. →
  брать из `NUM_PER_BATCH`/опции.
- **F.** Размер `QuerySessionPool` не настраивается. → опция `--session-pool-size`.
- **A.** Нет CLI-override метрики (не блокер — берётся из датасета).

---

## 6. Ускорение построения индекса

`ALTER TABLE … ADD INDEX vector_kmeans_tree` — фоновая операция, сканирующая всю
таблицу и строящая дерево кластеров. Время определяется K-means на каждом уровне
(`≈ N × clusters × dim × итерации`), записью posting table (а для покрывающего
индекса — копий векторов), множителем `overlap_clusters` и числом шардов
таблицы/узлов. Подходы по убыванию эффекта:

1. **Обучение на выборке.** Документация YDB допускает построение индекса на
   репрезентативной выборке (например, 10–50 % случайных строк) с последующей
   дозагрузкой остатка (новые строки распределяются по готовым кластерам).
   Дорогой итеративный K-means считается на `S ≪ N`, выигрыш ≈ `N/S`. Условия:
   не строить на пустой таблице; остаток грузить **после** завершения построения
   (во время построения обновления неконсистентны). Требует доработки загрузчика
   (двухфазная загрузка для YDB) — наибольший эффект на крупных датасетах.
2. **Предразбиение таблицы** (`PARTITION_AT_KEYS` по диапазону id) — скан
   построения идёт параллельно по шардам/узлам. Заодно ускоряет загрузку и
   снимает требование прогрева (см. §5 G′/H).
3. **Форма дерева.** Для типового железа документация рекомендует 20–50 кластеров
   при ≤512 векторах в листе (текущая авто-формула этому соответствует). Держать
   `clusters` ближе к нижней границе, не завышать `levels` (на ≲1–2M часто хватает
   2), полноту добирать через `overlap_clusters≈3` — это позволяет взять более
   дешёвое дерево без потери recall.
4. **Без покрытия.** Когда узкое место — именно время построения, строить без
   `COVER (embedding)` (`--no-cover-embedding`): построение и размер меньше ценой
   более медленного поиска.
5. **Инфраструктура + устранение D.** Больше CPU/узлов ускоряет K-means (в паре с
   предразбиением); устранение таймаута D не ускоряет, но позволяет построению
   завершиться на больших датасетах.

Быстрый реализуемый выигрыш — пункты 2 + 3 + устранение D без изменения
конвейера; пункт 1 — самый мощный, но требует доработки загрузчика.
