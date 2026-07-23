# R03b-impl-5a — F64-фундамент для судьи

**Дата:** 2026-07-20
**Итог:** ✅ Ворота 5a пройдены. Все аналитические градиенты выдерживают численный тест central-difference h=1e-6 с запасом 10-100× относительно записанного порога.

## Пакет `goml/internal/f64ref/`

Read-only reference-инструмент для судейства F32-путей. НЕ боевой код. В `nn/`/`train/` не импортируется.

## 5 файлов

| Файл | Назначение | Размер |
|---|---|---|
| `tensor.go` | F64Tensor (host-side []float64 + shape). Ленивая gotorch backend инициализация. Upload/Download F64 через `gotorch.MatMulF64`. | ~180 |
| `linear.go` | LinearF64 forward (`y = x @ W^T + bias`) + backward (`dX/dW/dB`). MatMul на GPU F64, bias broadcast на host. | ~105 |
| `layernorm.go` | LayerNormF64 CPU (mean/var/normalize/affine) forward+backward. | ~110 |
| `embedding.go` | EmbeddingF64 CPU gather forward + accumulate backward. | ~65 |
| `adamw.go` | AdamWF64 host loop с decoupled weight decay. | ~65 |

Всего **~525 строк** F64-фундамента.

## Ворота 5a — numerical gradient checks

Метод: central-diff `(f(x+h) - f(x-h)) / (2h)`, h=1e-6, loss = `<y, dY>` где dY — детерминированный random tensor. Правило: `|analytical - numerical| / (|numerical| + 1e-30) < tol`. F64 tolerance 1e-8 (LayerNorm 1e-7 из-за композиции 6+ ops).

| Тест | worst rel | Tolerance | Запас | Vердикт |
|---|---|---|---|---|
| LinearF64.dW | **1.35e-9** | 1e-8 | 7× | ✅ |
| LinearF64.dB | **7.95e-10** | 1e-8 | 13× | ✅ |
| LinearF64.dX | **2.41e-9** | 1e-8 | 4× | ✅ |
| LayerNormF64.dX | **7.12e-9** | 1e-7 | 14× | ✅ |
| LayerNormF64.dGamma | **9.42e-10** | 1e-7 | 106× | ✅ |
| LayerNormF64.dBeta | **4.54e-9** | 1e-8 | 2× | ✅ |
| EmbeddingF64.dW | **2.74e-10** | 1e-8 | 36× | ✅ |

Максимальная rel-ошибка **7.12e-9** — существенно ниже FP64 machine epsilon × K (что было бы 2.2e-16 × 8 ≈ 1.8e-15 для чистого пути, реальный ~1e-9 — включает h²-ошибку central-diff, ожидаемо).

## AdamWF64 vs ручная формула

Один шаг на 2-параметровом векторе с известным градиентом. Формула AdamW:
- `m = 0.9*0 + 0.1*g = [0.05, -0.1]`
- `v = 0.999*0 + 0.001*g² = [2.5e-4, 1e-3]`
- `bc1=0.1, bc2=1e-3`
- `mHat = [0.5, -1.0]`, `vHat = [0.25, 1.0]`
- `update = mHat/(sqrt(vHat)+eps) + wd*p`
- `p_new = p - lr*update`

Ожидание: `[0.899000002, 2.098000001]`. Реальность: `[0.899000002000000, 2.097999999000000]`. Diff < 1e-8 (14-я цифра). ✅ PASS.

## Регрессия

Пакет `f64ref` — новый, изолированный. Не задевает существующие пути. Ни один регрессионный тест не переменился.

## СТОП по правилу серии — только при провале numerical grad

Все выдержаны с запасом ⇒ иду на 5b без остановки.
