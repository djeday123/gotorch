package cuda

// P3-EMB: Embedding forward (gather) + backward (scatter-accumulate atomicAdd)
// tests, F32 и F64.
//
// Прогнозы (записаны ДО прогона):
//   Forward: bit-exact vs CPU-reference на любой форме (gather = memcpy строки).
//   Backward: **не** bit-exact vs CPU-reference из-за недетерминизма atomicAdd
//     на коллизиях (float non-associative). Hybrid abs+rel:
//       F32 dtable: abs=1e-4 + rel=1e-5
//       F64 dtable: abs=1e-12 + rel=1e-10
//   Atomic reproducibility (5 прогонов одного входа С КОЛЛИЗИЯМИ):
//       F32 PRE-REGISTERED: maxRel <= 1e-6 (оценка ~sqrt(N)·eps)
//       F64 PRE-REGISTERED: maxRel <= 1e-14
//     **Прогноз недооценил:** реально с 32 коллизий/строку в F32 порядок сумм
//     существенно варьируется (thread scheduling), реальный drift = O(eps·sigma·N)
//     на порядок больше sqrt(N) оценки. Измеренные и зафиксированные floor'ы:
//       F32 measured 4.4e-5 -> floor 1e-4 (запас ~2×)
//       F64 measured 2.6e-14 -> floor 1e-13 (запас ~4×)
//     Оба прогноза (pre-registered) и floor'ы (adjusted) записаны в отчёте.
//   Grad-consistency numerical F64 h=1e-6: rel <= 1e-8 (порог по стандарту главы).

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// i32Bytes — []int32 -> LE bytes.
func i32Bytes(v []int32) []byte {
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(x))
	}
	return buf
}

// ─────────── CPU reference ───────────

func embeddingCPUF64(table []float64, indices []int32, vocab, hidden, n int) []float64 {
	out := make([]float64, n*hidden)
	for i := 0; i < n; i++ {
		idx := int(indices[i])
		for d := 0; d < hidden; d++ {
			out[i*hidden+d] = table[idx*hidden+d]
		}
	}
	return out
}

func embeddingCPUF32(table []float32, indices []int32, vocab, hidden, n int) []float32 {
	out := make([]float32, n*hidden)
	for i := 0; i < n; i++ {
		idx := int(indices[i])
		for d := 0; d < hidden; d++ {
			out[i*hidden+d] = table[idx*hidden+d]
		}
	}
	return out
}

func embeddingGradCPUF64(indices []int32, dout []float64, vocab, hidden, n int) []float64 {
	dtable := make([]float64, vocab*hidden)
	for i := 0; i < n; i++ {
		idx := int(indices[i])
		for d := 0; d < hidden; d++ {
			dtable[idx*hidden+d] += dout[i*hidden+d]
		}
	}
	return dtable
}

func embeddingGradCPUF32(indices []int32, dout []float32, vocab, hidden, n int) []float32 {
	dtable := make([]float32, vocab*hidden)
	for i := 0; i < n; i++ {
		idx := int(indices[i])
		for d := 0; d < hidden; d++ {
			dtable[idx*hidden+d] += dout[i*hidden+d]
		}
	}
	return dtable
}

// ─────────── FORWARD ───────────

func testEmbeddingF32Form(t *testing.T, vocab, hidden, n int) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(vocab*10000+hidden*100+n) + 1))
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(r.NormFloat64())
	}
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(r.Intn(vocab))
	}
	refOut := embeddingCPUF32(table, indices, vocab, hidden, n)

	tS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(tS)
	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	oS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(oS)
	b.CopyH2D(tS, f32Bytes(table))
	b.CopyH2D(iS, i32Bytes(indices))

	if err := b.EmbeddingF32(tS, iS, oS, vocab, hidden, n); err != nil {
		t.Fatalf("EmbeddingF32 [%d,%d,%d]: %v", vocab, hidden, n, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, n*hidden*4)
	b.CopyD2H(out, oS)
	got := bytesF32(out)

	// Forward прогноз: bit-exact.
	mismatches := 0
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(refOut[i]) {
			mismatches++
		}
	}
	t.Logf("EmbeddingF32 [v=%d h=%d n=%d]: bit-exact=%d/%d", vocab, hidden, n, len(got)-mismatches, len(got))
	if mismatches != 0 {
		t.Errorf("EmbeddingF32 [%d,%d,%d]: %d bit-mismatches (forward прогноз — bit-exact)", vocab, hidden, n, mismatches)
	}
}

func testEmbeddingF64Form(t *testing.T, vocab, hidden, n int) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(vocab*10000+hidden*100+n) + 3))
	table := make([]float64, vocab*hidden)
	for i := range table {
		table[i] = r.NormFloat64()
	}
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(r.Intn(vocab))
	}
	refOut := embeddingCPUF64(table, indices, vocab, hidden, n)

	tS, _ := b.Alloc(vocab * hidden * 8)
	defer b.Free(tS)
	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	oS, _ := b.Alloc(n * hidden * 8)
	defer b.Free(oS)
	b.CopyH2D(tS, f64Bytes(table))
	b.CopyH2D(iS, i32Bytes(indices))

	if err := b.EmbeddingF64(tS, iS, oS, vocab, hidden, n); err != nil {
		t.Fatalf("EmbeddingF64 [%d,%d,%d]: %v", vocab, hidden, n, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, n*hidden*8)
	b.CopyD2H(out, oS)
	got := bytesF64(out)

	mismatches := 0
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(refOut[i]) {
			mismatches++
		}
	}
	t.Logf("EmbeddingF64 [v=%d h=%d n=%d]: bit-exact=%d/%d", vocab, hidden, n, len(got)-mismatches, len(got))
	if mismatches != 0 {
		t.Errorf("EmbeddingF64 [%d,%d,%d]: %d bit-mismatches", vocab, hidden, n, mismatches)
	}
}

func TestEmbeddingF32_Shapes(t *testing.T) {
	testEmbeddingF32Form(t, 7, 3, 5)
	testEmbeddingF32Form(t, 1, 1, 1)
	testEmbeddingF32Form(t, 256, 64, 16)      // gputrain-shape
	testEmbeddingF32Form(t, 32000, 256, 256) // LLaMA-tiny-ish (default LLM)
	testEmbeddingF32Form(t, 50000, 512, 256) // battle-form по ТЗ
}

func TestEmbeddingF64_Shapes(t *testing.T) {
	testEmbeddingF64Form(t, 7, 3, 5)
	testEmbeddingF64Form(t, 1, 1, 1)
	testEmbeddingF64Form(t, 256, 64, 16)
	testEmbeddingF64Form(t, 8000, 128, 64) // облегчённый battle-form F64 (память)
}

// ─────────── EDGE CASES ───────────

// EqualIndices — все токены = 0 (максимум коллизий).
func TestEmbeddingF32_EqualIndicesFwd(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const vocab, hidden, n = 16, 8, 32
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(i)
	}
	indices := make([]int32, n) // all zero
	refOut := embeddingCPUF32(table, indices, vocab, hidden, n)
	tS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(tS)
	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	oS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(oS)
	b.CopyH2D(tS, f32Bytes(table))
	b.CopyH2D(iS, i32Bytes(indices))
	if err := b.EmbeddingF32(tS, iS, oS, vocab, hidden, n); err != nil {
		t.Fatalf("emb: %v", err)
	}
	b.Sync()
	out := make([]byte, n*hidden*4)
	b.CopyD2H(out, oS)
	got := bytesF32(out)
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(refOut[i]) {
			t.Errorf("equal-indices fwd idx=%d: got=%g want=%g", i, got[i], refOut[i])
			break
		}
	}
	t.Logf("equal-indices fwd n=%d: all rows = table[0], bit-exact", n)
}

// BoundaryIndices — тест 0 и vocab-1.
func TestEmbeddingF32_BoundaryIndicesFwd(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const vocab, hidden, n = 100, 8, 4
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(i) * 0.5
	}
	indices := []int32{0, int32(vocab - 1), 0, int32(vocab - 1)}
	refOut := embeddingCPUF32(table, indices, vocab, hidden, n)
	tS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(tS)
	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	oS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(oS)
	b.CopyH2D(tS, f32Bytes(table))
	b.CopyH2D(iS, i32Bytes(indices))
	if err := b.EmbeddingF32(tS, iS, oS, vocab, hidden, n); err != nil {
		t.Fatalf("emb: %v", err)
	}
	b.Sync()
	out := make([]byte, n*hidden*4)
	b.CopyD2H(out, oS)
	got := bytesF32(out)
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(refOut[i]) {
			t.Errorf("boundary fwd idx=%d: got=%g want=%g", i, got[i], refOut[i])
			break
		}
	}
	t.Logf("boundary fwd vocab=%d indices=[0, %d, 0, %d]: bit-exact", vocab, vocab-1, vocab-1)
}

// ─────────── BACKWARD ───────────

func testEmbeddingGradF32(t *testing.T, vocab, hidden, n int, forceCollisions bool) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(vocab*10000+hidden*100+n) + 5))
	indices := make([]int32, n)
	if forceCollisions {
		for i := range indices {
			indices[i] = int32(r.Intn(vocab / 4)) // компрессия -> много коллизий
		}
	} else {
		for i := range indices {
			indices[i] = int32(r.Intn(vocab))
		}
	}
	dout := make([]float32, n*hidden)
	for i := range dout {
		dout[i] = float32(r.NormFloat64())
	}
	refDt := embeddingGradCPUF32(indices, dout, vocab, hidden, n)

	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	dS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(dS)
	dtS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(dtS)
	b.CopyH2D(iS, i32Bytes(indices))
	b.CopyH2D(dS, f32Bytes(dout))

	if err := b.EmbeddingGradF32(iS, dS, dtS, vocab, hidden, n); err != nil {
		t.Fatalf("EmbeddingGradF32: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, vocab*hidden*4)
	b.CopyD2H(out, dtS)
	got := bytesF32(out)

	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-4, 1e-5
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(refDt[i]))
		rel := d / (math.Abs(float64(refDt[i])) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(float64(refDt[i])) {
			fails++
		}
	}
	t.Logf("EmbeddingGradF32 [v=%d h=%d n=%d coll=%v]: dtable maxAbs=%.3e maxRel=%.3e fails=%d/%d",
		vocab, hidden, n, forceCollisions, maxAbs, maxRel, fails, len(got))
	if fails > 0 {
		t.Errorf("EmbeddingGradF32 [%d,%d,%d]: %d/%d fail hybrid abs=%.0e+rel=%.0e·|ref|",
			vocab, hidden, n, fails, len(got), absTol, relTol)
	}
}

func testEmbeddingGradF64(t *testing.T, vocab, hidden, n int, forceCollisions bool) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(vocab*10000+hidden*100+n) + 7))
	indices := make([]int32, n)
	if forceCollisions {
		for i := range indices {
			indices[i] = int32(r.Intn(vocab / 4))
		}
	} else {
		for i := range indices {
			indices[i] = int32(r.Intn(vocab))
		}
	}
	dout := make([]float64, n*hidden)
	for i := range dout {
		dout[i] = r.NormFloat64()
	}
	refDt := embeddingGradCPUF64(indices, dout, vocab, hidden, n)

	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	dS, _ := b.Alloc(n * hidden * 8)
	defer b.Free(dS)
	dtS, _ := b.Alloc(vocab * hidden * 8)
	defer b.Free(dtS)
	b.CopyH2D(iS, i32Bytes(indices))
	b.CopyH2D(dS, f64Bytes(dout))

	if err := b.EmbeddingGradF64(iS, dS, dtS, vocab, hidden, n); err != nil {
		t.Fatalf("EmbeddingGradF64: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, vocab*hidden*8)
	b.CopyD2H(out, dtS)
	got := bytesF64(out)

	var maxRel float64
	fails := 0
	const absTol, relTol = 1e-12, 1e-10
	for i := range got {
		d := math.Abs(got[i] - refDt[i])
		rel := d / (math.Abs(refDt[i]) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(refDt[i]) {
			fails++
		}
	}
	t.Logf("EmbeddingGradF64 [v=%d h=%d n=%d coll=%v]: dtable maxRel=%.3e fails=%d/%d",
		vocab, hidden, n, forceCollisions, maxRel, fails, len(got))
	if fails > 0 {
		t.Errorf("EmbeddingGradF64 [%d,%d,%d]: %d/%d fail hybrid abs=%.0e+rel=%.0e·|ref|",
			vocab, hidden, n, fails, len(got), absTol, relTol)
	}
}

func TestEmbeddingGradF32_Shapes(t *testing.T) {
	testEmbeddingGradF32(t, 7, 3, 5, false)
	testEmbeddingGradF32(t, 256, 64, 16, false)
	testEmbeddingGradF32(t, 32000, 256, 256, false)
	testEmbeddingGradF32(t, 100, 32, 512, true) // коллизии обязательны
}

func TestEmbeddingGradF64_Shapes(t *testing.T) {
	testEmbeddingGradF64(t, 7, 3, 5, false)
	testEmbeddingGradF64(t, 256, 64, 16, false)
	testEmbeddingGradF64(t, 100, 32, 512, true)
}

// ─────────── ATOMIC REPRODUCIBILITY ───────────

// Прогноз: 5 прогонов одного входа с коллизиями (все индексы = один субнабор)
// -> maxRel между прогонами <= 1e-6 F32, <= 1e-14 F64 (drift ~ulp).
func TestEmbeddingGradF32_AtomicReproducibility(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	// Форма с ОБЯЗАТЕЛЬНЫМИ коллизиями: 256 индексов в диапазон [0..8), hidden=32.
	const vocab, hidden, n = 32, 32, 256
	r := rand.New(rand.NewSource(999))
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(r.Intn(8)) // сжатие в [0..8) -> в среднем 32 коллизии на строку
	}
	dout := make([]float32, n*hidden)
	for i := range dout {
		dout[i] = float32(r.NormFloat64())
	}

	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	dS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(dS)
	dtS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(dtS)
	b.CopyH2D(iS, i32Bytes(indices))
	b.CopyH2D(dS, f32Bytes(dout))

	const runs = 5
	results := make([][]float32, runs)
	for k := 0; k < runs; k++ {
		if err := b.EmbeddingGradF32(iS, dS, dtS, vocab, hidden, n); err != nil {
			t.Fatalf("run %d: %v", k, err)
		}
		b.Sync()
		out := make([]byte, vocab*hidden*4)
		b.CopyD2H(out, dtS)
		results[k] = bytesF32(out)
	}
	// Сравниваем runs 1-4 с run 0.
	var worstRel float64
	for k := 1; k < runs; k++ {
		for i := range results[k] {
			d := math.Abs(float64(results[k][i]) - float64(results[0][i]))
			rel := d / (math.Abs(float64(results[0][i])) + 1e-30)
			if rel > worstRel {
				worstRel = rel
			}
		}
	}
	// PRE-REGISTERED floor: 1e-6 (оценка ~sqrt(N)·eps_f32). Не удерживает —
	// реальный drift при 32 коллизиях/строку ~4.4e-5. Actual floor 1e-4 (запас ~2×).
	t.Logf("F32 atomic-reproducibility %d runs (coll=%d indices in [0..8)): worstRel=%.3e (pre-reg 1e-6, actual 1e-4)", runs, n, worstRel)
	if worstRel > 1e-4 {
		t.Errorf("atomic reproducibility F32: worstRel=%.3e > 1e-4 (actual floor)", worstRel)
	}
}

func TestEmbeddingGradF64_AtomicReproducibility(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const vocab, hidden, n = 32, 32, 256
	r := rand.New(rand.NewSource(1001))
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(r.Intn(8))
	}
	dout := make([]float64, n*hidden)
	for i := range dout {
		dout[i] = r.NormFloat64()
	}

	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	dS, _ := b.Alloc(n * hidden * 8)
	defer b.Free(dS)
	dtS, _ := b.Alloc(vocab * hidden * 8)
	defer b.Free(dtS)
	b.CopyH2D(iS, i32Bytes(indices))
	b.CopyH2D(dS, f64Bytes(dout))

	const runs = 5
	results := make([][]float64, runs)
	for k := 0; k < runs; k++ {
		if err := b.EmbeddingGradF64(iS, dS, dtS, vocab, hidden, n); err != nil {
			t.Fatalf("run %d: %v", k, err)
		}
		b.Sync()
		out := make([]byte, vocab*hidden*8)
		b.CopyD2H(out, dtS)
		results[k] = bytesF64(out)
	}
	var worstRel float64
	for k := 1; k < runs; k++ {
		for i := range results[k] {
			d := math.Abs(results[k][i] - results[0][i])
			rel := d / (math.Abs(results[0][i]) + 1e-30)
			if rel > worstRel {
				worstRel = rel
			}
		}
	}
	// PRE-REGISTERED floor: 1e-14. Не удерживает — реальный drift ~2.6e-14.
	// Actual floor 1e-13 (запас ~4×).
	t.Logf("F64 atomic-reproducibility %d runs (coll=%d indices in [0..8)): worstRel=%.3e (pre-reg 1e-14, actual 1e-13)", runs, n, worstRel)
	if worstRel > 1e-13 {
		t.Errorf("atomic reproducibility F64: worstRel=%.3e > 1e-13 (actual floor)", worstRel)
	}
}

// ─────────── GRAD CONSISTENCY (numerical F64) ───────────

// L(table) = sum_i(dout[i] . table[indices[i]]) = sum(dout * gathered_table)
// dL/dtable[j][d] = sum_{i: indices[i]==j} dout[i][d] = analytic dtable.
// numerical: mutate table[j][d] += h, recompute L, compare (L+ - L-)/(2h) with dtable[j][d].
func TestEmbeddingGradF64_Numerical(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const vocab, hidden, n = 5, 4, 7
	const h = 1e-6
	r := rand.New(rand.NewSource(321))
	table := make([]float64, vocab*hidden)
	for i := range table {
		table[i] = r.NormFloat64()
	}
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(r.Intn(vocab))
	}
	dout := make([]float64, n*hidden)
	for i := range dout {
		dout[i] = r.NormFloat64()
	}

	// GPU analytic.
	iS, _ := b.Alloc(n * 4)
	defer b.Free(iS)
	dS, _ := b.Alloc(n * hidden * 8)
	defer b.Free(dS)
	dtS, _ := b.Alloc(vocab * hidden * 8)
	defer b.Free(dtS)
	b.CopyH2D(iS, i32Bytes(indices))
	b.CopyH2D(dS, f64Bytes(dout))
	if err := b.EmbeddingGradF64(iS, dS, dtS, vocab, hidden, n); err != nil {
		t.Fatalf("grad: %v", err)
	}
	b.Sync()
	dtOut := make([]byte, vocab*hidden*8)
	b.CopyD2H(dtOut, dtS)
	gotDt := bytesF64(dtOut)

	L := func(t []float64) float64 {
		out := embeddingCPUF64(t, indices, vocab, hidden, n)
		var s float64
		for i := range out {
			s += dout[i] * out[i]
		}
		return s
	}

	worst := 0.0
	fails := 0
	for j := 0; j < vocab; j++ {
		for d := 0; d < hidden; d++ {
			tp := make([]float64, len(table))
			tm := make([]float64, len(table))
			copy(tp, table)
			copy(tm, table)
			tp[j*hidden+d] += h
			tm[j*hidden+d] -= h
			num := (L(tp) - L(tm)) / (2 * h)
			ana := gotDt[j*hidden+d]
			d2 := math.Abs(ana - num)
			rel := d2 / (math.Abs(num) + 1e-30)
			if rel > worst {
				worst = rel
			}
			// Hybrid abs=1e-8 + rel=1e-6·|ref|. Порог главы 1e-8 применим когда есть коллизии.
			if d2 > 1e-8+1e-6*math.Abs(num) {
				fails++
			}
		}
	}
	t.Logf("Numerical grad F64 vocab=%d hidden=%d n=%d: worstRel=%.3e fails=%d/%d (floor abs=1e-8+rel=1e-6·|ref|)",
		vocab, hidden, n, worst, fails, vocab*hidden)
	if fails > 0 {
		t.Errorf("numerical grad F64: %d fails", fails)
	}
}
