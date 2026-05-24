package tensor

import (
	"math"
	"testing"
)

// ─── #7 Welford variance ────────────────────────────────────────────────────

func TestVarianceWelfordStability(t *testing.T) {
	// Build a sequence with a huge offset so naïve (x-mean)^2 catastrophically
	// cancels in single precision, but Welford remains numerically stable.
	// True variance of {1,2,3,4,5} is 2.5 (unbiased N-1). Adding 1e9 shouldn't
	// change it.
	const offset = 1e9
	data := []float64{
		1 + offset,
		2 + offset,
		3 + offset,
		4 + offset,
		5 + offset,
	}
	tt := New(data, []int{5})
	v := Var(tt, -999, false, true)
	got := v.Item()
	if math.Abs(got-2.5) > 1e-6 {
		t.Errorf("Welford variance with offset %g: got %v, want 2.5", offset, got)
	}
}

func TestVarianceMatchesPyTorch(t *testing.T) {
	// Cross-check unbiased variance with offset-free baseline.
	tt := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int{10})
	got := Var(tt, -999, false, true).Item()
	want := 9.166666666666666 // unbiased variance of [1..10]
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("Var [1..10]: got %v, want %v", got, want)
	}
}

// ─── #8 TopK heap ───────────────────────────────────────────────────────────

func TestTopKBasic(t *testing.T) {
	tt := New([]float64{3, 1, 4, 1, 5, 9, 2, 6}, []int{8})
	vals, idxs := TopK(tt, 3)
	wantVals := []float64{9, 6, 5}
	wantIdxs := []float64{5, 7, 4}
	for i := range wantVals {
		if vals.Data()[i] != wantVals[i] {
			t.Errorf("TopK val %d: got %v, want %v", i, vals.Data()[i], wantVals[i])
		}
		if idxs.Data()[i] != wantIdxs[i] {
			t.Errorf("TopK idx %d: got %v, want %v", i, idxs.Data()[i], wantIdxs[i])
		}
	}
}

func TestTopKAllElements(t *testing.T) {
	// k == n: should return all elements sorted desc.
	tt := New([]float64{2, 5, 1, 4, 3}, []int{5})
	vals, _ := TopK(tt, 5)
	want := []float64{5, 4, 3, 2, 1}
	for i := range want {
		if vals.Data()[i] != want[i] {
			t.Errorf("TopK %d: got %v, want %v", i, vals.Data()[i], want[i])
		}
	}
}

func BenchmarkTopK_LargeN_SmallK(b *testing.B) {
	// 100k elements, k=50 — old O(n*k) would do 5M ops per call.
	data := make([]float64, 100_000)
	for i := range data {
		data[i] = float64((i*1234567)%97 - 48)
	}
	tt := New(data, []int{100_000})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TopK(tt, 50)
	}
}

// ─── #10 In-place ops ───────────────────────────────────────────────────────

func TestAddInPlace(t *testing.T) {
	a := Zeros(3)
	a.data[0], a.data[1], a.data[2] = 1, 2, 3
	b := New([]float64{10, 20, 30}, []int{3})
	a.AddInPlace(b)
	want := []float64{11, 22, 33}
	for i, v := range a.Data() {
		if v != want[i] {
			t.Errorf("AddInPlace[%d]: got %v, want %v", i, v, want[i])
		}
	}
}

func TestMulScalarInPlace(t *testing.T) {
	a := New([]float64{1, 2, 3, 4}, []int{4})
	a.MulScalarInPlace(2.5)
	want := []float64{2.5, 5, 7.5, 10}
	for i, v := range a.Data() {
		if v != want[i] {
			t.Errorf("MulScalarInPlace[%d]: got %v, want %v", i, v, want[i])
		}
	}
}

func TestInPlaceShapeMismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	a := Zeros(3)
	b := Zeros(4)
	a.AddInPlace(b)
}

func BenchmarkAddVsAddInPlace(b *testing.B) {
	x := New(make([]float64, 1024), []int{1024})
	y := New(make([]float64, 1024), []int{1024})
	b.Run("Add_alloc", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Add(x, y)
		}
	})
	b.Run("AddInPlace", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			x.AddInPlace(y)
		}
	})
}

// ─── #9 Pool round-trip ─────────────────────────────────────────────────────

func TestPoolRoundTrip(t *testing.T) {
	// Allocate, fill, return, re-allocate same size — must be zeroed again.
	s := AllocFloat64(100)
	for i := range s {
		s[i] = float64(i)
	}
	FreeFloat64(s)
	s2 := AllocFloat64(100)
	for i, v := range s2 {
		if v != 0 {
			t.Errorf("pool returned non-zeroed slice at [%d] = %v", i, v)
		}
	}
}

func TestTensorReleaseSafe(t *testing.T) {
	// Release a tensor and confirm nothing panics in subsequent zero-alloc.
	tt := Zeros(64, 64)
	tt.Release()
	tt2 := Zeros(64, 64)
	if tt2.Size() != 64*64 {
		t.Errorf("post-release alloc size: %d", tt2.Size())
	}
}
