package tensor

import (
	"math"
	"testing"
)

// ── no_grad is tested in autograd package ──

// ── float32 ──────────────────────────────────────────────────────────────

func TestFloat32Creation(t *testing.T) {
	a := ZerosF32(2, 3)
	if a.DType() != Float32 {
		t.Fatal("expected Float32 dtype")
	}
	if a.Shape()[0] != 2 || a.Shape()[1] != 3 {
		t.Fatal("wrong shape")
	}
	for _, v := range a.DataF32() {
		if v != 0 {
			t.Fatal("expected zeros")
		}
	}
}

func TestFloat32Ops(t *testing.T) {
	a := NewF32([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewF32([]float32{1, 1, 1, 1}, []int{2, 2})

	c := Add(a, b)
	if c.DType() != Float32 {
		t.Fatal("Add float32+float32 should return Float32")
	}
	want := []float32{2, 3, 4, 5}
	got := c.DataF32()
	for i, v := range want {
		if math.Abs(float64(got[i]-v)) > 1e-6 {
			t.Errorf("index %d: got %v want %v", i, got[i], v)
		}
	}
}

func TestFloat32Cast(t *testing.T) {
	f64 := New([]float64{1.5, 2.5, 3.5}, []int{3})
	f32 := f64.Float32()
	if f32.DType() != Float32 {
		t.Fatal("expected Float32")
	}
	back := f32.Float64()
	if back.DType() != Float64 {
		t.Fatal("expected Float64")
	}
	for i, v := range []float64{1.5, 2.5, 3.5} {
		if math.Abs(back.Data()[i]-v) > 1e-4 {
			t.Errorf("index %d: got %v want %v", i, back.Data()[i], v)
		}
	}
}

func TestFloat32Item(t *testing.T) {
	s := NewF32([]float32{3.14}, []int{1})
	if math.Abs(s.Item()-3.14) > 1e-4 {
		t.Fatalf("Item() = %v, want ~3.14", s.Item())
	}
}

func TestFloat32RandN(t *testing.T) {
	t_ := RandNF32(100)
	if t_.DType() != Float32 {
		t.Fatal("expected Float32")
	}
	if t_.Size() != 100 {
		t.Fatalf("expected 100 elements, got %d", t_.Size())
	}
}

// ── cat ───────────────────────────────────────────────────────────────────

func TestCat1D(t *testing.T) {
	a := New([]float64{1, 2, 3}, []int{3})
	b := New([]float64{4, 5}, []int{2})
	c := Cat([]*Tensor{a, b}, 0)
	if c.Shape()[0] != 5 {
		t.Fatalf("expected shape [5], got %v", c.Shape())
	}
	want := []float64{1, 2, 3, 4, 5}
	for i, v := range want {
		if c.Data()[i] != v {
			t.Errorf("index %d: got %v want %v", i, c.Data()[i], v)
		}
	}
}

func TestCat2DDim0(t *testing.T) {
	a := New([]float64{1, 2, 3, 4}, []int{2, 2})
	b := New([]float64{5, 6, 7, 8, 9, 10}, []int{3, 2})
	c := Cat([]*Tensor{a, b}, 0)
	if c.Shape()[0] != 5 || c.Shape()[1] != 2 {
		t.Fatalf("wrong shape: %v", c.Shape())
	}
}

func TestCat2DDim1(t *testing.T) {
	a := New([]float64{1, 2, 3, 4}, []int{2, 2})
	b := New([]float64{5, 6, 7, 8}, []int{2, 2})
	c := Cat([]*Tensor{a, b}, 1)
	if c.Shape()[0] != 2 || c.Shape()[1] != 4 {
		t.Fatalf("wrong shape: %v", c.Shape())
	}
	if c.At(0, 2) != 5 || c.At(1, 3) != 8 {
		t.Errorf("wrong values: %v", c.Data())
	}
}

// ── stack ─────────────────────────────────────────────────────────────────

func TestStack(t *testing.T) {
	a := New([]float64{1, 2, 3}, []int{3})
	b := New([]float64{4, 5, 6}, []int{3})
	c := Stack([]*Tensor{a, b}, 0)
	if c.Shape()[0] != 2 || c.Shape()[1] != 3 {
		t.Fatalf("wrong shape: %v", c.Shape())
	}
	if c.At(0, 0) != 1 || c.At(1, 2) != 6 {
		t.Errorf("wrong values")
	}
}

func TestStackDim1(t *testing.T) {
	a := New([]float64{1, 2}, []int{2})
	b := New([]float64{3, 4}, []int{2})
	c := Stack([]*Tensor{a, b}, 1) // shape [2, 2]
	if c.Shape()[0] != 2 || c.Shape()[1] != 2 {
		t.Fatalf("wrong shape: %v", c.Shape())
	}
}

// ── split / chunk ─────────────────────────────────────────────────────────

func TestSplit(t *testing.T) {
	a := Arange(0, 10, 1)
	chunks := Split(a, 3, 0)
	if len(chunks) != 4 { // 3+3+3+1
		t.Fatalf("expected 4 chunks, got %d", len(chunks))
	}
	if chunks[0].Size() != 3 || chunks[3].Size() != 1 {
		t.Errorf("wrong chunk sizes: %d, %d", chunks[0].Size(), chunks[3].Size())
	}
}

func TestChunk(t *testing.T) {
	a := Arange(0, 9, 1)
	chunks := Chunk(a, 3, 0)
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}
	for _, c := range chunks {
		if c.Size() != 3 {
			t.Errorf("expected chunk size 3, got %d", c.Size())
		}
	}
}

// ── indexing ──────────────────────────────────────────────────────────────

func TestSelect(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	row := a.Select(0, 1) // second row
	if row.Shape()[0] != 3 {
		t.Fatalf("expected shape [3], got %v", row.Shape())
	}
	if row.At(0) != 4 || row.At(1) != 5 || row.At(2) != 6 {
		t.Errorf("wrong values: %v", row.Data())
	}
}

func TestNarrow(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5}, []int{5})
	b := a.Narrow(0, 1, 3) // [2, 3, 4]
	if b.Shape()[0] != 3 {
		t.Fatalf("expected shape [3], got %v", b.Shape())
	}
	if b.At(0) != 2 || b.At(2) != 4 {
		t.Errorf("wrong values: %v", b.Data())
	}
}

func TestIndex(t *testing.T) {
	a := New([]float64{10, 20, 30, 40, 50}, []int{5})
	b := a.Index([]int{4, 1, 0})
	if b.Shape()[0] != 3 {
		t.Fatalf("expected shape [3], got %v", b.Shape())
	}
	if b.At(0) != 50 || b.At(1) != 20 || b.At(2) != 10 {
		t.Errorf("wrong values: %v", b.Data())
	}
}

func TestMaskedSelect(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5}, []int{5})
	mask := New([]float64{1, 0, 1, 0, 1}, []int{5})
	b := a.MaskedSelect(mask)
	if b.Size() != 3 {
		t.Fatalf("expected 3 elements, got %d", b.Size())
	}
	if b.At(0) != 1 || b.At(1) != 3 || b.At(2) != 5 {
		t.Errorf("wrong values: %v", b.Data())
	}
}

func TestClamp(t *testing.T) {
	a := New([]float64{-2, 0, 3, 5}, []int{4})
	b := Clamp(a, 0, 4)
	want := []float64{0, 0, 3, 4}
	for i, v := range want {
		if b.Data()[i] != v {
			t.Errorf("index %d: got %v want %v", i, b.Data()[i], v)
		}
	}
}

func TestWhere(t *testing.T) {
	cond := New([]float64{1, 0, 1, 0}, []int{4})
	a := New([]float64{10, 20, 30, 40}, []int{4})
	b := New([]float64{1, 2, 3, 4}, []int{4})
	c := Where(cond, a, b)
	want := []float64{10, 2, 30, 4}
	for i, v := range want {
		if c.Data()[i] != v {
			t.Errorf("index %d: got %v want %v", i, c.Data()[i], v)
		}
	}
}
