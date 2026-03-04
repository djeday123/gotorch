package tensor

import "fmt"

// ---------------------------------------------------------------------------
// SparseCOO — sparse tensor in COO (Coordinate) format
// ---------------------------------------------------------------------------
//
// A SparseCOO stores only non-zero values and their indices.
// Equivalent to torch.sparse_coo_tensor.
//
//	indices := [][]int{{0, 1}, {1, 2}}   // 2 non-zeros at (0,1) and (1,2)
//	values  := []float64{3.0, 4.0}
//	s := tensor.NewSparseCOO([]int{3, 4}, indices, values)
type SparseCOO struct {
	Shape   []int
	Indices [][]int   // len = nnz, each element is an nd-index
	Values  []float64 // len = nnz
}

// NewSparseCOO creates a sparse COO tensor.
// shape: dense shape, indices: list of nd-indices, values: corresponding values.
func NewSparseCOO(shape []int, indices [][]int, values []float64) *SparseCOO {
	if len(indices) != len(values) {
		panic("SparseCOO: len(indices) != len(values)")
	}
	return &SparseCOO{Shape: shape, Indices: indices, Values: values}
}

// NNZ returns the number of non-zero elements.
func (s *SparseCOO) NNZ() int { return len(s.Values) }

// ToDense converts the sparse tensor to a dense *Tensor.
func (s *SparseCOO) ToDense() *Tensor {
	t := Zeros(s.Shape...)
	for i, idx := range s.Indices {
		t.Set(s.Values[i], idx...)
	}
	return t
}

// String returns a compact representation.
func (s *SparseCOO) String() string {
	return fmt.Sprintf("SparseCOO(shape=%v, nnz=%d)", s.Shape, s.NNZ())
}

// SparseMM multiplies sparse matrix A [M, K] by dense matrix B [K, N] → dense [M, N].
// Equivalent to torch.sparse.mm(A, B).
func SparseMM(a *SparseCOO, b *Tensor) *Tensor {
	if len(a.Shape) != 2 {
		panic("SparseMM: a must be 2D")
	}
	M, K := a.Shape[0], a.Shape[1]
	bShape := b.Shape()
	if bShape[0] != K {
		panic(fmt.Sprintf("SparseMM: shape mismatch: a[%d,%d] @ b[%d,%d]", M, K, bShape[0], bShape[1]))
	}
	N := bShape[1]

	out := Zeros(M, N)
	bData := b.Data()

	for i, idx := range a.Indices {
		row, col := idx[0], idx[1]
		v := a.Values[i]
		for n := 0; n < N; n++ {
			cur := out.At(row, n)
			out.Set(cur+v*bData[col*N+n], row, n)
		}
	}
	return out
}

// SparseAdd adds a sparse tensor to a dense tensor element-wise.
// Returns a new dense tensor. Both must have identical shapes.
func SparseAdd(sparse *SparseCOO, dense *Tensor) *Tensor {
	// Copy dense data into a new tensor
	src := dense.Data()
	dst := make([]float64, len(src))
	copy(dst, src)
	result := New(dst, dense.Shape())
	for i, idx := range sparse.Indices {
		cur := result.At(idx...)
		result.Set(cur+sparse.Values[i], idx...)
	}
	return result
}

// ToDense alias as a free function for convenience.
func SparseToDense(s *SparseCOO) *Tensor { return s.ToDense() }
