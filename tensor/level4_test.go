package tensor

import (
	"math"
	"testing"
)

// ── Full & Linspace ──────────────────────────────────────────────────────────

func TestFull(t *testing.T) {
	f := Full(3.14, 2, 3)
	if f.Shape()[0] != 2 || f.Shape()[1] != 3 {
		t.Fatalf("shape %v", f.Shape())
	}
	for i, v := range f.Data() {
		if v != 3.14 {
			t.Errorf("index %d: got %v want 3.14", i, v)
		}
	}
}

func TestLinspace(t *testing.T) {
	l := Linspace(0, 1, 5)
	want := []float64{0, 0.25, 0.5, 0.75, 1.0}
	for i, v := range l.Data() {
		if math.Abs(v-want[i]) > 1e-12 {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

func TestLinspaceSingle(t *testing.T) {
	l := Linspace(5, 10, 1)
	if l.Data()[0] != 5 {
		t.Errorf("got %v want 5", l.Data()[0])
	}
}

// ── Floor / Ceil / Round / Sign ──────────────────────────────────────────────

func TestFloor(t *testing.T) {
	t1 := New([]float64{1.7, -1.2, 0.0}, []int{3})
	out := Floor(t1)
	want := []float64{1, -2, 0}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

func TestCeil(t *testing.T) {
	t1 := New([]float64{1.2, -1.7, 0.0}, []int{3})
	out := Ceil(t1)
	want := []float64{2, -1, 0}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

func TestRound(t *testing.T) {
	t1 := New([]float64{1.5, 2.5, -1.5, 0.4}, []int{4})
	out := Round(t1)
	// math.Round rounds half away from zero
	want := []float64{2, 3, -2, 0}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

func TestSign(t *testing.T) {
	t1 := New([]float64{3.0, -2.0, 0.0}, []int{3})
	out := Sign(t1)
	want := []float64{1, -1, 0}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

// ── Prod ──────────────────────────────────────────────────────────────────────

func TestProdAll(t *testing.T) {
	t1 := New([]float64{1, 2, 3, 4}, []int{4})
	got := Prod(t1, -999, false).Item()
	if got != 24 {
		t.Errorf("got %v want 24", got)
	}
}

func TestProdDim(t *testing.T) {
	t1 := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	out := Prod(t1, 1, false)
	want := []float64{6, 120}
	for i, v := range out.Data() {
		if math.Abs(v-want[i]) > 1e-10 {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

// ── Var & Std ─────────────────────────────────────────────────────────────────

func TestVar(t *testing.T) {
	// data: [2, 4, 4, 4] mean=3.5, var_unbiased = 0.5*((2-3.5)^2+(4-3.5)^2+...)
	t1 := New([]float64{2, 4, 4, 4}, []int{4})
	v := Var(t1, -999, false, true).Item()
	// Expected: ((2-3.5)^2+(4-3.5)^2*3) / 3 = (2.25+0.75)/3 = 1.0
	if math.Abs(v-1.0) > 1e-10 {
		t.Errorf("var=%v want 1.0", v)
	}
}

func TestStd(t *testing.T) {
	t1 := New([]float64{2, 4, 4, 4}, []int{4})
	s := Std(t1, -999, false, true).Item()
	if math.Abs(s-1.0) > 1e-10 {
		t.Errorf("std=%v want 1.0", s)
	}
}

func TestVarBiased(t *testing.T) {
	t1 := New([]float64{2, 4}, []int{2})
	v := Var(t1, -999, false, false).Item() // divide by N=2
	// mean=3, var=((2-3)^2+(4-3)^2)/2=1
	if math.Abs(v-1.0) > 1e-10 {
		t.Errorf("var biased=%v want 1.0", v)
	}
}

// ── Norm ─────────────────────────────────────────────────────────────────────

func TestNormL2(t *testing.T) {
	t1 := New([]float64{3, 4}, []int{2})
	n := Norm(t1, 2).Item()
	if math.Abs(n-5.0) > 1e-10 {
		t.Errorf("L2 norm = %v want 5.0", n)
	}
}

func TestNormL1(t *testing.T) {
	t1 := New([]float64{-3, 4}, []int{2})
	n := Norm(t1, 1).Item()
	if math.Abs(n-7.0) > 1e-10 {
		t.Errorf("L1 norm = %v want 7.0", n)
	}
}

func TestNormInf(t *testing.T) {
	t1 := New([]float64{-3, 1, 4}, []int{3})
	n := Norm(t1, math.Inf(1)).Item()
	if math.Abs(n-4.0) > 1e-10 {
		t.Errorf("Linf norm = %v want 4.0", n)
	}
}

// ── Cumsum ────────────────────────────────────────────────────────────────────

func TestCumsum(t *testing.T) {
	t1 := New([]float64{1, 2, 3, 4}, []int{4})
	out := Cumsum(t1, 0)
	want := []float64{1, 3, 6, 10}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

func TestCumsumMatrix(t *testing.T) {
	// [2,3] cumsum along dim=1
	t1 := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	out := Cumsum(t1, 1)
	want := []float64{1, 3, 6, 4, 9, 15}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}

// ── TopK ─────────────────────────────────────────────────────────────────────

func TestTopK(t *testing.T) {
	t1 := New([]float64{3, 1, 4, 1, 5, 9, 2, 6}, []int{8})
	vals, idxs := TopK(t1, 3)
	// top 3: 9 (idx 5), 6 (idx 7), 5 (idx 4)
	wantVals := []float64{9, 6, 5}
	wantIdxs := []float64{5, 7, 4}
	for i := range wantVals {
		if vals.Data()[i] != wantVals[i] {
			t.Errorf("vals[%d]=%v want %v", i, vals.Data()[i], wantVals[i])
		}
		if idxs.Data()[i] != wantIdxs[i] {
			t.Errorf("idxs[%d]=%v want %v", i, idxs.Data()[i], wantIdxs[i])
		}
	}
}

// ── NormDim ───────────────────────────────────────────────────────────────────

func TestNormDim(t *testing.T) {
	// [[3,4],[0,1]] L2 along dim=1 → [5, 1]
	t1 := New([]float64{3, 4, 0, 1}, []int{2, 2})
	out := NormDim(t1, 2, 1, false)
	want := []float64{5, 1}
	for i, v := range out.Data() {
		if math.Abs(v-want[i]) > 1e-10 {
			t.Errorf("[%d]: got %v want %v", i, v, want[i])
		}
	}
}
