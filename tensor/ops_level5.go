package tensor

import "fmt"

// flatToMulti converts a flat index to multi-dimensional indices given a shape.
func flatToMulti(flatIdx int, shape []int) []int {
	ndim := len(shape)
	multi := make([]int, ndim)
	for d := ndim - 1; d >= 0; d-- {
		multi[d] = flatIdx % shape[d]
		flatIdx /= shape[d]
	}
	return multi
}

// Gather gathers values along dim using index.
// output[i0,...,dim_idx,...,in] = input[i0,...,index[i0,...,dim_idx,...,in],...,in]
// index must have the same number of dimensions as t.
func (t *Tensor) Gather(dim int, index *Tensor) *Tensor {
	ndim := t.Ndim()
	if dim < 0 {
		dim += ndim
	}
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("tensor.Gather: dim %d out of range [0, %d)", dim, ndim))
	}

	idxShape := index.Shape()
	out := Zeros(idxShape...)
	outSize := totalSize(idxShape)
	idxData := index.Data()

	for i := 0; i < outSize; i++ {
		multiIdx := flatToMulti(i, idxShape)
		srcIdx := make([]int, ndim)
		copy(srcIdx, multiIdx)
		srcIdx[dim] = int(idxData[i])
		out.data[i] = t.At(srcIdx...)
	}
	return out
}

// ScatterAdd adds values from src into a copy of t along dim using index.
// output starts as a copy of t, then: output[..., index[i,j,k], ...] += src[i,j,k]
// index and src must have the same shape.
func (t *Tensor) ScatterAdd(dim int, index, src *Tensor) *Tensor {
	ndim := t.Ndim()
	if dim < 0 {
		dim += ndim
	}
	out := t.ContiguousCopy()
	idxShape := index.Shape()
	idxSize := totalSize(idxShape)
	idxData := index.Data()
	srcData := src.Data()

	for i := 0; i < idxSize; i++ {
		multiIdx := flatToMulti(i, idxShape)
		dstIdx := make([]int, ndim)
		copy(dstIdx, multiIdx)
		dstIdx[dim] = int(idxData[i])
		flatDst := out.flatIndex(dstIdx)
		out.data[flatDst] += srcData[i]
	}
	return out
}

// Cumsum computes the cumulative sum along dim.
func (t *Tensor) Cumsum(dim int) *Tensor {
	shape := t.Shape()
	ndim := len(shape)
	if dim < 0 {
		dim += ndim
	}
	out := t.ContiguousCopy()

	outerSize := 1
	for d := 0; d < dim; d++ {
		outerSize *= shape[d]
	}
	dimLen := shape[dim]
	innerSize := 1
	for d := dim + 1; d < ndim; d++ {
		innerSize *= shape[d]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			sum := 0.0
			for k := 0; k < dimLen; k++ {
				flatIdx := outer*dimLen*innerSize + k*innerSize + inner
				sum += out.data[flatIdx]
				out.data[flatIdx] = sum
			}
		}
	}
	return out
}

// Cumprod computes the cumulative product along dim.
func (t *Tensor) Cumprod(dim int) *Tensor {
	shape := t.Shape()
	ndim := len(shape)
	if dim < 0 {
		dim += ndim
	}
	out := t.ContiguousCopy()

	outerSize := 1
	for d := 0; d < dim; d++ {
		outerSize *= shape[d]
	}
	dimLen := shape[dim]
	innerSize := 1
	for d := dim + 1; d < ndim; d++ {
		innerSize *= shape[d]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			prod := 1.0
			for k := 0; k < dimLen; k++ {
				flatIdx := outer*dimLen*innerSize + k*innerSize + inner
				prod *= out.data[flatIdx]
				out.data[flatIdx] = prod
			}
		}
	}
	return out
}

// Tril returns the lower triangular part of a 2D matrix (rows × cols).
// diagonal=0  → keep main diagonal and below
// diagonal>0  → keep up to `diagonal` diagonals above main
// diagonal<0  → exclude `|diagonal|` diagonals below main
func (t *Tensor) Tril(diagonal int) *Tensor {
	shape := t.Shape()
	if len(shape) != 2 {
		panic("tensor.Tril: only 2D tensors supported")
	}
	rows, cols := shape[0], shape[1]
	out := t.ContiguousCopy()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j > i+diagonal {
				out.data[i*cols+j] = 0
			}
		}
	}
	return out
}

// Triu returns the upper triangular part of a 2D matrix.
// diagonal=0  → keep main diagonal and above
// diagonal>0  → keep only `diagonal` diagonals above main
// diagonal<0  → include `|diagonal|` diagonals below main
func (t *Tensor) Triu(diagonal int) *Tensor {
	shape := t.Shape()
	if len(shape) != 2 {
		panic("tensor.Triu: only 2D tensors supported")
	}
	rows, cols := shape[0], shape[1]
	out := t.ContiguousCopy()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j < i+diagonal {
				out.data[i*cols+j] = 0
			}
		}
	}
	return out
}

// RepeatInterleave repeats each element `repeats` times along dim.
// E.g. [1,2,3].RepeatInterleave(2, 0) → [1,1,2,2,3,3]
func (t *Tensor) RepeatInterleave(repeats int, dim int) *Tensor {
	shape := t.Shape()
	ndim := len(shape)
	if dim < 0 {
		dim += ndim
	}
	outShape := make([]int, ndim)
	copy(outShape, shape)
	outShape[dim] = shape[dim] * repeats

	out := Zeros(outShape...)
	outSize := totalSize(outShape)

	for i := 0; i < outSize; i++ {
		multiIdx := flatToMulti(i, outShape)
		srcIdx := make([]int, ndim)
		copy(srcIdx, multiIdx)
		srcIdx[dim] = multiIdx[dim] / repeats
		out.data[i] = t.At(srcIdx...)
	}
	return out
}
