package tensor

import (
	"math"
	"testing"
)

func TestSparseCOOToDense(t *testing.T) {
	// 3x3 sparse with values at (0,1)=2.0 and (2,0)=5.0
	s := NewSparseCOO(
		[]int{3, 3},
		[][]int{{0, 1}, {2, 0}},
		[]float64{2.0, 5.0},
	)
	dense := s.ToDense()

	if math.Abs(dense.At(0, 1)-2.0) > 1e-10 {
		t.Errorf("(0,1): expected 2.0, got %v", dense.At(0, 1))
	}
	if math.Abs(dense.At(2, 0)-5.0) > 1e-10 {
		t.Errorf("(2,0): expected 5.0, got %v", dense.At(2, 0))
	}
	if math.Abs(dense.At(1, 1)) > 1e-10 {
		t.Errorf("(1,1): expected 0, got %v", dense.At(1, 1))
	}
}

func TestSparseCOONNZ(t *testing.T) {
	s := NewSparseCOO(
		[]int{5, 5},
		[][]int{{0, 0}, {1, 2}, {4, 4}},
		[]float64{1, 2, 3},
	)
	if s.NNZ() != 3 {
		t.Errorf("NNZ: expected 3, got %d", s.NNZ())
	}
}

func TestSparseCOOEmpty(t *testing.T) {
	s := NewSparseCOO([]int{4, 4}, nil, nil)
	dense := s.ToDense()
	for _, v := range dense.Data() {
		if v != 0 {
			t.Error("empty sparse should produce all-zero dense")
		}
	}
}

func TestSparseCOOString(t *testing.T) {
	s := NewSparseCOO([]int{3, 3}, [][]int{{0, 0}}, []float64{1.0})
	str := s.String()
	if str == "" {
		t.Error("String() should not be empty")
	}
}

func TestSparseMM(t *testing.T) {
	// A (3x3) sparse: identity-ish
	// A[0,0]=1, A[1,1]=2, A[2,2]=3
	a := NewSparseCOO(
		[]int{3, 3},
		[][]int{{0, 0}, {1, 1}, {2, 2}},
		[]float64{1, 2, 3},
	)
	// B (3x2) dense
	b := New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2})

	out := SparseMM(a, b)
	// row 0: 1*[1,2] = [1,2]
	// row 1: 2*[3,4] = [6,8]
	// row 2: 3*[5,6] = [15,18]
	expected := []float64{1, 2, 6, 8, 15, 18}
	got := out.Data()
	for i, v := range expected {
		if math.Abs(got[i]-v) > 1e-10 {
			t.Errorf("SparseMM[%d]: expected %v, got %v", i, v, got[i])
		}
	}
}

func TestSparseMMLarger(t *testing.T) {
	// A (2x4) sparse, B (4x3) dense
	a := NewSparseCOO(
		[]int{2, 4},
		[][]int{{0, 1}, {0, 3}, {1, 0}, {1, 2}},
		[]float64{1, 2, 3, 4},
	)
	b := New([]float64{
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,
		0, 1, 0,
	}, []int{4, 3})

	out := SparseMM(a, b)
	// row 0: 1*b[1] + 2*b[3] = [0,1,0] + [0,2,0] = [0,3,0]
	// row 1: 3*b[0] + 4*b[2] = [3,0,3] + [4,0,4] = [7,0,7]
	expected := []float64{0, 3, 0, 7, 0, 7}
	got := out.Data()
	for i, v := range expected {
		if math.Abs(got[i]-v) > 1e-10 {
			t.Errorf("SparseMM larger[%d]: expected %v, got %v", i, v, got[i])
		}
	}
}

func TestSparseAdd(t *testing.T) {
	s := NewSparseCOO(
		[]int{2, 3},
		[][]int{{0, 0}, {1, 2}},
		[]float64{10, 20},
	)
	d := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	out := SparseAdd(s, d)

	// (0,0): 10+1=11, (1,2): 20+6=26, rest unchanged
	if math.Abs(out.At(0, 0)-11) > 1e-10 {
		t.Errorf("(0,0): expected 11, got %v", out.At(0, 0))
	}
	if math.Abs(out.At(1, 2)-26) > 1e-10 {
		t.Errorf("(1,2): expected 26, got %v", out.At(1, 2))
	}
	if math.Abs(out.At(0, 1)-2) > 1e-10 {
		t.Errorf("(0,1): expected 2 unchanged, got %v", out.At(0, 1))
	}
}

func TestSparseAddDoesNotMutateDense(t *testing.T) {
	s := NewSparseCOO([]int{2, 2}, [][]int{{0, 0}}, []float64{99})
	d := New([]float64{1, 2, 3, 4}, []int{2, 2})
	_ = SparseAdd(s, d)
	if d.At(0, 0) != 1 {
		t.Error("SparseAdd should not mutate input dense tensor")
	}
}

func TestSparseToDense(t *testing.T) {
	s := NewSparseCOO([]int{2, 2}, [][]int{{1, 1}}, []float64{7})
	dense := SparseToDense(s)
	if math.Abs(dense.At(1, 1)-7) > 1e-10 {
		t.Errorf("SparseToDense(1,1): expected 7, got %v", dense.At(1, 1))
	}
}
