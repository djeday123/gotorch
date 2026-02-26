package tensor

import (
	"math"
	"testing"
)

func almostEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

// --- Creation ---

func TestZeros(t *testing.T) {
	z := Zeros(2, 3)
	if z.Size() != 6 {
		t.Fatalf("expected size 6, got %d", z.Size())
	}
	for _, v := range z.data {
		if v != 0 {
			t.Fatal("expected all zeros")
		}
	}
}

func TestOnes(t *testing.T) {
	o := Ones(3, 3)
	for _, v := range o.data {
		if v != 1 {
			t.Fatal("expected all ones")
		}
	}
}

func TestEye(t *testing.T) {
	e := Eye(3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if e.At(i, j) != expected {
				t.Fatalf("Eye(%d,%d) = %f, want %f", i, j, e.At(i, j), expected)
			}
		}
	}
}

func TestArange(t *testing.T) {
	a := Arange(0, 5, 1)
	if a.Size() != 5 {
		t.Fatalf("expected 5 elements, got %d", a.Size())
	}
	for i := 0; i < 5; i++ {
		if a.At(i) != float64(i) {
			t.Fatalf("Arange[%d] = %f, want %f", i, a.At(i), float64(i))
		}
	}
}

// --- Indexing ---

func TestAtSet(t *testing.T) {
	m := Zeros(3, 4)
	m.Set(42.0, 1, 2)
	if m.At(1, 2) != 42.0 {
		t.Fatalf("expected 42.0, got %f", m.At(1, 2))
	}
}

// --- Shape ---

func TestReshape(t *testing.T) {
	a := Arange(0, 6, 1)
	b := a.Reshape(2, 3)
	if b.At(1, 1) != 4.0 {
		t.Fatalf("expected 4.0, got %f", b.At(1, 1))
	}
}

func TestReshapeInferDim(t *testing.T) {
	a := Arange(0, 12, 1)
	b := a.Reshape(3, -1)
	if b.shape[1] != 4 {
		t.Fatalf("expected inferred dim 4, got %d", b.shape[1])
	}
}

func TestTranspose(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := a.T()
	// b[0][1] should be a[1][0] = 4
	if b.At(0, 1) != 4.0 {
		t.Fatalf("expected 4.0, got %f", b.At(0, 1))
	}
	// b[2][0] = a[0][2] = 3
	if b.At(2, 0) != 3.0 {
		t.Fatalf("expected 3.0, got %f", b.At(2, 0))
	}
}

func TestFlatten(t *testing.T) {
	a := Ones(2, 3, 4)
	b := a.Flatten()
	if len(b.shape) != 1 || b.shape[0] != 24 {
		t.Fatalf("expected shape [24], got %v", b.shape)
	}
}

func TestSqueeze(t *testing.T) {
	a := Zeros(1, 3, 1, 4)
	b := a.Squeeze()
	if len(b.shape) != 2 || b.shape[0] != 3 || b.shape[1] != 4 {
		t.Fatalf("expected shape [3,4], got %v", b.shape)
	}
}

func TestUnsqueeze(t *testing.T) {
	a := Zeros(3, 4)
	b := a.Unsqueeze(0)
	if len(b.shape) != 3 || b.shape[0] != 1 {
		t.Fatalf("expected shape [1,3,4], got %v", b.shape)
	}
}

// --- Broadcast ops ---

func TestAddBroadcast(t *testing.T) {
	// (3,1) + (1,4) = (3,4)
	a := New([]float64{1, 2, 3}, []int{3, 1})
	b := New([]float64{10, 20, 30, 40}, []int{1, 4})
	c := Add(a, b)
	if len(c.shape) != 2 || c.shape[0] != 3 || c.shape[1] != 4 {
		t.Fatalf("expected shape [3,4], got %v", c.shape)
	}
	// c[2][3] = a[2][0] + b[0][3] = 3 + 40 = 43
	if c.At(2, 3) != 43.0 {
		t.Fatalf("expected 43.0, got %f", c.At(2, 3))
	}
}

func TestAddScalar(t *testing.T) {
	a := Ones(2, 2)
	b := AddScalar(a, 5)
	if b.At(0, 0) != 6 {
		t.Fatal("AddScalar failed")
	}
}

// --- Linalg ---

func TestMatMul(t *testing.T) {
	// 2x3 * 3x2 -> 2x2
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := New([]float64{7, 8, 9, 10, 11, 12}, []int{3, 2})
	c := MatMul(a, b)

	// Naive reference
	ref := make([]float64, 4)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			for k := 0; k < 3; k++ {
				ref[i*2+j] += a.At(i, k) * b.At(k, j)
			}
		}
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if !almostEqual(c.At(i, j), ref[i*2+j], 1e-9) {
				t.Fatalf("MatMul[%d,%d] = %f, want %f", i, j, c.At(i, j), ref[i*2+j])
			}
		}
	}
}

func TestDot(t *testing.T) {
	a := New([]float64{1, 2, 3}, []int{3})
	b := New([]float64{4, 5, 6}, []int{3})
	d := Dot(a, b)
	if d.Item() != 32.0 {
		t.Fatalf("Dot = %f, want 32.0", d.Item())
	}
}

func TestOuter(t *testing.T) {
	a := New([]float64{1, 2}, []int{2})
	b := New([]float64{3, 4, 5}, []int{3})
	o := Outer(a, b)
	if o.shape[0] != 2 || o.shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", o.shape)
	}
	if o.At(1, 2) != 10.0 {
		t.Fatalf("Outer[1,2] = %f, want 10.0", o.At(1, 2))
	}
}

// --- Reduce ---

func TestSum(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	s := Sum(a, -999, false)
	if s.Item() != 21.0 {
		t.Fatalf("Sum all = %f, want 21.0", s.Item())
	}
	rowSum := Sum(a, 1, false)
	if rowSum.At(0) != 6.0 || rowSum.At(1) != 15.0 {
		t.Fatalf("Sum(dim=1) = %v, want [6,15]", rowSum.data)
	}
}

func TestMean(t *testing.T) {
	a := New([]float64{1, 2, 3, 4}, []int{2, 2})
	m := Mean(a, -999, false)
	if !almostEqual(m.Item(), 2.5, 1e-9) {
		t.Fatalf("Mean = %f, want 2.5", m.Item())
	}
}

func TestSoftmax(t *testing.T) {
	a := New([]float64{1, 2, 3, 4}, []int{1, 4})
	s := Softmax(a, 1)
	// Each row should sum to 1
	rowSum := Sum(s, 1, false)
	if !almostEqual(rowSum.At(0), 1.0, 1e-9) {
		t.Fatalf("Softmax row sum = %f, want 1.0", rowSum.At(0))
	}
	// All values should be positive
	for _, v := range s.data {
		if v <= 0 {
			t.Fatal("Softmax produced non-positive value")
		}
	}
}

func TestSoftmaxNumericalStability(t *testing.T) {
	// Large values that would overflow with naive exp
	a := New([]float64{1000, 1001, 1002}, []int{1, 3})
	s := Softmax(a, 1)
	sum := Sum(s, 1, false)
	if !almostEqual(sum.At(0), 1.0, 1e-9) {
		t.Fatalf("Softmax(large values) sum = %f, want 1.0", sum.At(0))
	}
}

func TestReLU(t *testing.T) {
	a := New([]float64{-3, -1, 0, 1, 3}, []int{5})
	r := ReLU(a)
	expected := []float64{0, 0, 0, 1, 3}
	for i, v := range expected {
		if r.At(i) != v {
			t.Fatalf("ReLU[%d] = %f, want %f", i, r.At(i), v)
		}
	}
}

func TestArgMax(t *testing.T) {
	a := New([]float64{3, 1, 4, 1, 5, 9, 2, 6}, []int{2, 4})
	idx := ArgMax(a, 1)
	if idx.At(0) != 2.0 {
		t.Fatalf("ArgMax row 0 = %f, want 2.0", idx.At(0))
	}
	if idx.At(1) != 1.0 { // row1=[5,9,2,6], max=9 at idx 1
		t.Fatalf("ArgMax row 1 = %f, want 1.0", idx.At(1))
	}
}

func TestContiguousCopy(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := a.T()
	c := b.ContiguousCopy()
	// b is transposed (3,2); c should be contiguous
	if c.At(0, 0) != 1.0 || c.At(1, 0) != 2.0 {
		t.Fatalf("ContiguousCopy of transposed tensor failed: %v", c.data)
	}
}
