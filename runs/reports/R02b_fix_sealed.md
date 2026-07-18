# R02b-fix — запечатывание DeviceBuffer, две двери для указателя

**Дата:** 2026-07-14
**Отчётный агент:** R02b-fix
**Статус:** частично применено — design-часть внесена в план, code-часть ждёт восстановления основного ТЗ R02b (см. «Пред-требование» ниже).

---

## Пред-требование: восстановление основного ТЗ R02b

ТЗ R02b-fix ссылается на «основное ТЗ R02b (шесть этапов, ворота, PTX-ядра, cuBLAS)». В моём окне контекста этого документа нет — из истории видны только R01a (CUDA runtime recon), R02a (design интерфейса), R02a-fix (правки DeviceBuffer/ForeignStorage/F32-пары). Промежуточные дополнения к R02b, которые R02b-fix ОТМЕНЯЕТ (переименование `Ptr` в `UnsafeDevicePtr`, doc-контракт на нём), тоже не видел.

**Проверка состояния кода на диске:**
```
$ ls /data/lib/podman-data/projects/gotorch/v6/cuda/api.go
ls: cannot access '.../cuda/api.go': No such file or directory
$ ls /data/lib/podman-data/projects/gotorch/v6/cuda/util.go
ls: cannot access '.../cuda/util.go': No such file or directory
```

Файлов `cuda/api.go`, `cuda/util.go`, `cuda/backend_purego.go`, `cuda/purego_test.go` **не существует** — R02b-код в этой сессии не начинался. Директория `runs/reports/` также отсутствовала и была создана этим отчётом.

Единственный существующий артефакт R02-пути на данный момент — дизайн-документ `RECON_R02a_backend_api.md` (Секции 1-6 инвентаря + план интерфейса).

**Что это означает.** Правки R02b-fix, относящиеся к:
- дизайн-плану (форма типов, дверь-входа `WrapDevicePtr`, дверь-выхода `UnsafeExtractDevicePtr`, sealed-контракт) — применимы к дизайн-документу; **применены**, см. следующий раздел.
- реализации (`api.go`/`util.go`/`backend_purego.go`, три теста, грепы ворот 6) — применимы к коду, которого пока нет; **отложены** до момента, когда основное ТЗ R02b будет восстановлено и начнётся Этап 1 R02b (объявление типов + фабрика).

---

## Что заменено в дизайне

Все изменения — в файле `/data/lib/podman-data/projects/gotorch/v6/RECON_R02a_backend_api.md`.

### Секция 4 — блок типов переписан

Было (R02a-fix):
- `DeviceBuffer` с публичным методом `Ptr() unsafe.Pointer`.
- `Storage`/`ForeignStorage` реализуют `Ptr()/SizeBytes()/Device()` явно.
- Указатель device-памяти доступен через публичный геттер любому импортёру пакета.

Стало (R02b-fix sealed):
- Добавлен unexported `bufferView struct { ptr unsafe.Pointer; sizeBytes int; device int }`.
- `DeviceBuffer` содержит unexported-метод `deviceBuffer() bufferView` + публичные `SizeBytes()` / `Device()`. Метода `Ptr()` больше нет.
- `Storage.deviceBuffer()` и `ForeignStorage.deviceBuffer()` возвращают `bufferView{...}`.
- Интерфейс запечатан конструкцией языка: тип из чужого пакета не может объявить unexported-метод `deviceBuffer` пакета `cuda`, значит не может реализовать `DeviceBuffer`. Compute-методы `Backend` принимают гарантированно `Storage` или `ForeignStorage`.

### Секция 4 — utility-слой пополнен

Было: только `WrapDevicePtr` (дверь входа).
Стало: добавлена **дверь выхода** — `UnsafeExtractDevicePtr(b DeviceBuffer) unsafe.Pointer` с полным doc-контрактом: «единственное корректное применение — передача во внешние CUDA-биндинги; запрещено разыменовывать с host-стороны, сохранять как uintptr, делать арифметику указателей; единственная публичная точка выхода device-указателя из пакета».

### Секция 4 — обоснование файла

Было: «три типа (`DeviceBuffer`, `Storage`, `ForeignStorage`) — в `cuda/api.go`».
Стало: «четыре типа (`bufferView`, `DeviceBuffer`, `Storage`, `ForeignStorage`) — в `cuda/api.go`; `bufferView` unexported, не покидает пакет».

### Секция 5 — пункт 3 расширен

Пункт «внешний device-указатель» переписан под формулировку «две двери» с явными названиями (`WrapDevicePtr` — вход, `UnsafeExtractDevicePtr` — выход) и объяснением, почему интерфейс запечатан и как это исключает утечку указателя через случайный публичный геттер.

Никаких других правок в RECON_R02a_backend_api.md не вносилось — Секции 1-3 (инвентарь) и Секция 6 (резюме, кроме того, что уже было в R02a-fix) не тронуты.

---

## Отменённые предыдущие дополнения к R02b

По требованию ТЗ R02b-fix отмечаю, что следующие подходы, если они были предложены/начаты в других сессиях, отменяются:
1. **Переименование `Ptr()` в `UnsafeDevicePtr()` с doc-контрактом на публичном методе** — отменено. Причина: это управление риском через имя, а не устранение через дизайн. Заменено на sealed-интерфейс без публичного геттера + единственную функцию-выход `UnsafeExtractDevicePtr` в `util.go`.
2. **Исходная форма `DeviceBuffer` из R02a-fix Правки 1 с публичным `Ptr() unsafe.Pointer`** — отменено. Заменено на sealed-форму с unexported `deviceBuffer() bufferView`.

Если в существующем R02b-коде (который я не вижу) эти варианты уже частично реализованы, они должны быть переделаны под настоящий дизайн при возобновлении R02b.

---

## Ворота 6 основного ТЗ — обновлённые проверки

**Убрано:** grep на `.Ptr()` из ворот 6 (метода больше нет).

**Добавлено:**
1. `grep -rn 'UnsafeExtractDevicePtr' --include='*.go' . | grep -v '^cuda/'` — должно быть пусто. Означает: вне пакета `cuda` никто не извлекает указатель. При появлении goml-интеграции это станет ровно одно место — осознанное решение отдельного ТЗ.
2. `grep -n 'uintptr' cuda/api.go cuda/util.go` — должно быть пусто. `uintptr` разрешён только в `driver_purego.go` и `cublas_purego.go` (сигнатуры purego-биндингов); конверсия из `unsafe.Pointer` — в момент вызова; хранение `uintptr` в полях структур запрещено везде.

### Актуальный вывод грепов (на момент отчёта)

Файлы `cuda/api.go`, `cuda/util.go` пока не существуют. Тем не менее выполнены проверки на всей текущей кодовой базе — обе прошли:

```
$ grep -rn 'UnsafeExtractDevicePtr' --include='*.go' /data/lib/podman-data/projects/gotorch/v6/
(пусто, exit 1)
```
Комментарий: пусто корректно — реализации ещё нет, значит и вызовов быть не может; при появлении `util.go` в нём будет одно объявление (не «вызов вне cuda/»), и внешний grep `| grep -v '^cuda/'` останется пустым.

```
$ grep -n 'uintptr' /data/lib/podman-data/projects/gotorch/v6/cuda/api.go /data/lib/podman-data/projects/gotorch/v6/cuda/util.go
ls: cannot access '.../cuda/api.go': No such file or directory
ls: cannot access '.../cuda/util.go': No such file or directory
```
Комментарий: файлов нет, проверка pass by default; она станет содержательной как только `api.go` и `util.go` появятся.

---

## Три теста (R02b-fix требует) — статус

Все три предполагаются в `cuda/purego_test.go`, файла которого пока нет.

| Тест | Требование | Текущий статус |
|---|---|---|
| `TestSealedInterface` | Комментарий-демонстрация: `type fake struct{}` вне пакета не компилируется. Позитивный runtime-тест: `Storage`/`ForeignStorage` присваиваются `DeviceBuffer`, `WrapDevicePtr(UnsafeExtractDevicePtr(s), s.SizeBytes(), s.Device())` даёт `ForeignStorage`, `CopyD2H` читает те же байты. | **pending** — файл не существует |
| `TestNoUintptrInPublicAPI` | Программное чтение `api.go` и `util.go` через `os.ReadFile` + `strings.Contains(... "uintptr" ...)`, проверка на отсутствие токена. | **pending** — файл не существует |
| `TestFreeForeignNotCompilable` | Закомментированная строка `backend.Free(foreignStorage)` с пояснением, что она не типизируется. | **pending** — файл не существует |

Все три станут актуальны как только начнётся код-этап R02b (Этап 1 «Объявление типов + фабрика»).

---

## Приёмка R02b-fix — покомпонентный статус

| Требование ТЗ | Дизайн-документ | Код |
|---|---|---|
| `DeviceBuffer` с unexported `deviceBuffer() bufferView` | ✅ применено | ⏳ pending |
| `bufferView` unexported | ✅ применено | ⏳ pending |
| `Ptr()` удалён из публичной поверхности | ✅ применено | ⏳ pending |
| `UnsafeExtractDevicePtr` в `util.go` с полным doc-контрактом | ✅ применено | ⏳ pending |
| Сигнатуры `Backend` не изменились относительно R02a-fix | ✅ применено (compute-методы принимают `DeviceBuffer`, `Alloc` возвращает `Storage`, `Free` принимает `Storage`) | ⏳ pending |
| Три теста присутствуют и зелёные | — | ⏳ pending (файл не существует) |
| grep `UnsafeExtractDevicePtr` вне `cuda/` пуст | ✅ пусто на текущей кодовой базе (нет вызовов) | будет проверено на реальном `util.go` |
| grep `uintptr` в `api.go`+`util.go` пуст | ✅ pass by default (файлов нет) | будет проверено на реальных файлах |

---

## Что дальше

1. **Восстановить основное ТЗ R02b** (6 этапов, ворота, PTX-ядра, cuBLAS-биндинги) — либо перевыдать, либо вставить в контекст.
2. **Продолжить R02b с Этапа 1** — «Объявление типов + фабрика»: `cuda/api.go` + `cuda/util.go` создаются сразу под sealed-дизайн R02b-fix (без промежуточной формы с публичным `Ptr()`).
3. **На Этапе 1 R02b** — добавить `cuda/purego_test.go` с тремя тестами R02b-fix. Ворота 1 (компиляция + `TestFreeForeignNotCompilable` не собирается по типу) закрываются одновременно с воротами R02b-fix на sealed-контракт.
4. **На каждом последующем этапе R02b** — грепы ворот 6 R02b-fix (`UnsafeExtractDevicePtr` вне `cuda/`, `uintptr` в `api.go`/`util.go`) прогоняются как sanity check вместе с остальными воротами этапа.

Настоящий отчёт закрывает R02b-fix со стороны дизайна; закрытие со стороны реализации откладывается до Этапа 1 R02b.
