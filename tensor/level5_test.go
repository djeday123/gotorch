package tensor_test

import (
	"testing"

	"github.com/djeday123/gotorch/tensor"
)

// ── Gather ────────────────────────────────────────────────────────────────────

func TestGatherDim0(t *testing.T) {
	// input: [[1,2],[3,4],[5,6]]  shape [3,2]
	// index: [[0,2],[1,0]]        shape [2,2]
	// output[i,j] = input[index[i,j], j]
	inp := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2})
	idx := tensor.New([]float64{0, 2, 1, 0}, []int{2, 2})
	out := inp.Gather(0, idx)
	if out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Fatalf("Gather dim0 shape: want [2,2], got %v", out.Shape())
	}
	want := []float64{1, 6, 3, 2}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("Gather dim0 [%d]: want %v, got %v", i, v, got[i])
		}
	}
}

func TestGatherDim1(t *testing.T) {
	// input: [[1,2,3],[4,5,6]]  shape [2,3]
	// index: [[0,2],[1,0]]       shape [2,2]
	// output[i,j] = input[i, index[i,j]]
	inp := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	idx := tensor.New([]float64{0, 2, 1, 0}, []int{2, 2})
	out := inp.Gather(1, idx)
	if out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Fatalf("Gather dim1 shape: want [2,2], got %v", out.Shape())
	}
	want := []float64{1, 3, 5, 4}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("Gather dim1 [%d]: want %v, got %v", i, v, got[i])
		}
	}
}

// ── ScatterAdd ────────────────────────────────────────────────────────────────

func TestScatterAdd(t *testing.T) {
	// base: zeros [3,3]
	// index: [[0,1,2]] shape [1,3]
	// src:   [[1,2,3]] shape [1,3]
	// output[index[0,j], j] += src[0,j]  for dim=0
	base := tensor.Zeros(3, 3)
	idx := tensor.New([]float64{0, 1, 2}, []int{1, 3})
	src := tensor.New([]float64{1, 2, 3}, []int{1, 3})
	out := base.ScatterAdd(0, idx, src)
	// output[0,0]+=1, output[1,1]+=2, output[2,2]+=3
	if out.At(0, 0) != 1 {
		t.Errorf("ScatterAdd [0,0]: want 1, got %v", out.At(0, 0))
	}
	if out.At(1, 1) != 2 {
		t.Errorf("ScatterAdd [1,1]: want 2, got %v", out.At(1, 1))
	}
	if out.At(2, 2) != 3 {
		t.Errorf("ScatterAdd [2,2]: want 3, got %v", out.At(2, 2))
	}
	if out.At(0, 1) != 0 {
		t.Errorf("ScatterAdd [0,1]: want 0, got %v", out.At(0, 1))
	}
}

// ── Cumsum ────────────────────────────────────────────────────────────────────

func TestCumsumDim1(t *testing.T) {
	// [[1,2,3],[4,5,6]]  cumsum along dim=1
	inp := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	out := inp.Cumsum(1)
	want := []float64{1, 3, 6, 4, 9, 15}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("Cumsum[%d]: want %v, got %v", i, v, got[i])
		}
	}
}

func TestCumsumDim0(t *testing.T) {
	// [[1,2],[3,4],[5,6]] cumsum along dim=0
	inp := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2})
	out := inp.Cumsum(0)
	want := []float64{1, 2, 4, 6, 9, 12}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("Cumsum dim0[%d]: want %v, got %v", i, v, got[i])
		}
	}
}

// ── Cumprod ───────────────────────────────────────────────────────────────────

func TestCumprodDim1(t *testing.T) {
	// [[1,2,3],[1,2,4]] cumprod along dim=1
	inp := tensor.New([]float64{1, 2, 3, 1, 2, 4}, []int{2, 3})
	out := inp.Cumprod(1)
	want := []float64{1, 2, 6, 1, 2, 8}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("Cumprod[%d]: want %v, got %v", i, v, got[i])
		}
	}
}

// ── Tril ──────────────────────────────────────────────────────────────────────

func TestTril(t *testing.T) {
	inp := tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{3, 3})
	out := inp.Tril(0)
	// expected:
	// 1 0 0
	// 4 5 0
	// 7 8 9
	expected := []float64{1, 0, 0, 4, 5, 0, 7, 8, 9}
	got := out.Data()
	for i, v := range expected {
		if got[i] != v {
			t.Errorf("Tril[%d]: want %v, got %v", i, v, got[i])
		}
	}
}

func TestTrilPositiveDiag(t *testing.T) {
	inp := tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{3, 3})
	out := inp.Tril(1)
	// keep up to 1 above main diagonal:
	// 1 2 0
	// 4 5 6
	// 7 8 9
	expected := []float64{1, 2, 0, 4, 5, 6, 7, 8, 9}
	got := out.Data()
	for i, v := range expected {
		if got[i] != v {
			t.Errorf("TrilDiag1[%d]: want %v, got %v", i, v, got[i])
		}
	}
}

// ── Triu ──────────────────────────────────────────────────────────────────────

func TestTriu(t *testing.T) {
	inp := tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{3, 3})
	out := inp.Triu(0)
	// expected:
	// 1 2 3
	// 0 5 6
	// 0 0 9
	expected := []float64{1, 2, 3, 0, 5, 6, 0, 0, 9}
	got := out.Data()
	for i, v := range expected {
		if got[i] != v {
			t.Errorf("Triu[%d]: want %v, got %v", i, v, got[i])
		}
	}
}

// ── RepeatInterleave ──────────────────────────────────────────────────────────

func TestRepeatInterleaveDim0(t *testing.T) {
	// [1,2,3] repeat 2 along dim=0 → [1,1,2,2,3,3]
	inp := tensor.New([]float64{1, 2, 3}, []int{3})
	out := inp.RepeatInterleave(2, 0)
	if out.Shape()[0] != 6 {
		t.Fatalf("RepeatInterleave shape: want [6], got %v", out.Shape())
	}
	want := []float64{1, 1, 2, 2, 3, 3}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("RepeatInterleave[%d]: want %v, got %v", i, v, got[i])
		}
	}
}

func TestRepeatInterleave2D(t *testing.T) {
	// [[1,2],[3,4]] repeat 3 along dim=1 → [[1,1,1,2,2,2],[3,3,3,4,4,4]]
	inp := tensor.New([]float64{1, 2, 3, 4}, []int{2, 2})
	out := inp.RepeatInterleave(3, 1)
	if out.Shape()[0] != 2 || out.Shape()[1] != 6 {
		t.Fatalf("RepeatInterleave2D shape: want [2,6], got %v", out.Shape())
	}
	want := []float64{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("RepeatInterleave2D[%d]: want %v, got %v", i, v, got[i])
		}
	}
}
