package tensor

import "fmt"

const tileSize = 32

// MatMul performs matrix multiplication using a tiled algorithm for cache efficiency.
// Supports 2D tensors. For batched matmul use BatchMatMul.
//
// dtype: if both inputs are Float32 the result is Float32; otherwise Float64.
// Internal accumulation is in Float64 to keep numerical stability across K.
func MatMul(a, b *Tensor) *Tensor {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic(fmt.Sprintf("tensor: MatMul expects 2D tensors, got %dD and %dD", len(a.shape), len(b.shape)))
	}
	M, K := a.shape[0], a.shape[1]
	K2, N := b.shape[0], b.shape[1]
	if K != K2 {
		panic(fmt.Sprintf("tensor: MatMul shape mismatch: (%d,%d) x (%d,%d)", M, K, K2, N))
	}

	isF32 := a.dtype == Float32 && b.dtype == Float32

	// Make contiguous copies, promoting to Float64 if necessary so the inner
	// loop has a single data path. Conversion is cheap compared to the
	// matmul itself.
	ac := a.ContiguousCopy()
	bc := b.ContiguousCopy()
	if ac.dtype == Float32 {
		ac = ac.Float64()
	}
	if bc.dtype == Float32 {
		bc = bc.Float64()
	}
	out := Zeros(M, N)

	// Tiled matmul
	for i0 := 0; i0 < M; i0 += tileSize {
		for k0 := 0; k0 < K; k0 += tileSize {
			for j0 := 0; j0 < N; j0 += tileSize {
				iEnd := min(i0+tileSize, M)
				kEnd := min(k0+tileSize, K)
				jEnd := min(j0+tileSize, N)
				for i := i0; i < iEnd; i++ {
					for k := k0; k < kEnd; k++ {
						aik := ac.data[i*K+k]
						for j := j0; j < jEnd; j++ {
							out.data[i*N+j] += aik * bc.data[k*N+j]
						}
					}
				}
			}
		}
	}
	if isF32 {
		return out.Float32()
	}
	return out
}

// BatchMatMul performs batched matrix multiplication. Supports 3D tensors (batch, M, K) x (batch, K, N).
// dtype: Float32 if both inputs are Float32, otherwise Float64.
func BatchMatMul(a, b *Tensor) *Tensor {
	if len(a.shape) != 3 || len(b.shape) != 3 {
		panic("tensor: BatchMatMul expects 3D tensors")
	}
	batch, M, K := a.shape[0], a.shape[1], a.shape[2]
	if b.shape[0] != batch || b.shape[1] != K {
		panic(fmt.Sprintf("tensor: BatchMatMul shape mismatch: (%d,%d,%d) x (%d,%d,%d)", batch, M, K, b.shape[0], b.shape[1], b.shape[2]))
	}
	N := b.shape[2]
	isF32 := a.dtype == Float32 && b.dtype == Float32

	// Promote to Float64 internally — slice2D works on the data []float64
	// field directly, so it doesn't support Float32 storage.
	if a.dtype == Float32 {
		a = a.Float64()
	}
	if b.dtype == Float32 {
		b = b.Float64()
	}

	out := Zeros(batch, M, N)
	for i := 0; i < batch; i++ {
		ai := a.Reshape(a.shape[0], a.shape[1]*a.shape[2]).slice2D(i)
		bi := b.Reshape(b.shape[0], b.shape[1]*b.shape[2]).slice2D(i)
		ai = ai.Reshape(M, K)
		bi = bi.Reshape(K, N)
		res := MatMul(ai, bi)
		copy(out.data[i*M*N:], res.data)
	}
	if isF32 {
		return out.Float32()
	}
	return out
}

// slice2D returns row i of a 2D tensor as a 1D tensor (view).
func (t *Tensor) slice2D(i int) *Tensor {
	if len(t.shape) != 2 {
		panic("slice2D: need 2D tensor")
	}
	cols := t.shape[1]
	return &Tensor{
		data:    t.data,
		shape:   []int{1, cols},
		strides: []int{cols, 1},
		offset:  t.offset + i*cols,
	}
}

// Dot computes the dot product of two 1-D tensors.
func Dot(a, b *Tensor) *Tensor {
	if len(a.shape) != 1 || len(b.shape) != 1 {
		panic("tensor: Dot expects 1D tensors")
	}
	if a.shape[0] != b.shape[0] {
		panic(fmt.Sprintf("tensor: Dot size mismatch: %d vs %d", a.shape[0], b.shape[0]))
	}
	var sum float64
	ita, itb := newIterator(a), newIterator(b)
	for ita.hasNext() {
		sum += a.data[ita.next()] * b.data[itb.next()]
	}
	return Scalar(sum)
}

// Outer computes the outer product of two 1-D tensors → 2D tensor (len(a), len(b)).
func Outer(a, b *Tensor) *Tensor {
	if len(a.shape) != 1 || len(b.shape) != 1 {
		panic("tensor: Outer expects 1D tensors")
	}
	m, n := a.shape[0], b.shape[0]
	out := Zeros(m, n)
	ita := newIterator(a)
	for i := 0; ita.hasNext(); i++ {
		av := a.data[ita.next()]
		itb := newIterator(b)
		for j := 0; itb.hasNext(); j++ {
			out.data[i*n+j] = av * b.data[itb.next()]
		}
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
