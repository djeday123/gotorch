package tensor

import "fmt"

// Cat concatenates tensors along dim.
// All tensors must have identical shape except along dim.
func Cat(tensors []*Tensor, dim int) *Tensor {
	if len(tensors) == 0 {
		panic("tensor.Cat: empty list")
	}
	if len(tensors) == 1 {
		return tensors[0].ContiguousCopy()
	}

	ref := tensors[0]
	ndim := ref.Ndim()
	if dim < 0 {
		dim += ndim
	}
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("tensor.Cat: dim %d out of range for %d-D tensor", dim, ndim))
	}

	// Validate shapes
	outDimSize := 0
	for _, t := range tensors {
		if t.Ndim() != ndim {
			panic("tensor.Cat: all tensors must have the same number of dimensions")
		}
		for d := 0; d < ndim; d++ {
			if d != dim && t.shape[d] != ref.shape[d] {
				panic(fmt.Sprintf("tensor.Cat: shape mismatch at dim %d: %d vs %d", d, t.shape[d], ref.shape[d]))
			}
		}
		outDimSize += t.shape[dim]
	}

	// Build output shape
	outShape := make([]int, ndim)
	copy(outShape, ref.shape)
	outShape[dim] = outDimSize

	out := Zeros(outShape...)
	offset := 0
	for _, t := range tensors {
		ct := t.ContiguousCopy()
		copyAlongDim(out, ct, dim, offset)
		offset += t.shape[dim]
	}
	return out
}

// copyAlongDim copies src into dst starting at position startIdx along dim.
func copyAlongDim(dst, src *Tensor, dim, startIdx int) {
	srcSize := src.shape[dim]
	// Iterate over all "slices" in the non-dim dimensions
	// We'll use a flat index approach with stride info
	ndim := dst.Ndim()
	outerSize := 1
	for d := 0; d < dim; d++ {
		outerSize *= dst.shape[d]
	}
	innerSize := 1
	for d := dim + 1; d < ndim; d++ {
		innerSize *= dst.shape[d]
	}
	dstDimStride := innerSize * dst.shape[dim] // not used but for clarity
	_ = dstDimStride

	for outer := 0; outer < outerSize; outer++ {
		for i := 0; i < srcSize; i++ {
			for inner := 0; inner < innerSize; inner++ {
				srcFlat := outer*srcSize*innerSize + i*innerSize + inner
				dstFlat := outer*dst.shape[dim]*innerSize + (startIdx+i)*innerSize + inner
				dst.data[dstFlat] = src.data[srcFlat]
			}
		}
	}
}

// Stack creates a new dimension and stacks tensors along it.
// All tensors must have identical shape. dim=0 stacks along a new first axis.
func Stack(tensors []*Tensor, dim int) *Tensor {
	if len(tensors) == 0 {
		panic("tensor.Stack: empty list")
	}
	ref := tensors[0]
	ndim := ref.Ndim()
	if dim < 0 {
		dim += ndim + 1
	}
	if dim < 0 || dim > ndim {
		panic(fmt.Sprintf("tensor.Stack: dim %d out of range", dim))
	}
	for _, t := range tensors {
		if t.Ndim() != ndim {
			panic("tensor.Stack: all tensors must have the same shape")
		}
		for d := 0; d < ndim; d++ {
			if t.shape[d] != ref.shape[d] {
				panic(fmt.Sprintf("tensor.Stack: shape mismatch at dim %d", d))
			}
		}
	}

	// Unsqueeze each tensor along dim, then cat
	unsqueezed := make([]*Tensor, len(tensors))
	for i, t := range tensors {
		unsqueezed[i] = t.Unsqueeze(dim)
	}
	return Cat(unsqueezed, dim)
}

// Split splits t into chunks of size splitSize along dim.
// The last chunk may be smaller if the dimension is not divisible.
func Split(t *Tensor, splitSize int, dim int) []*Tensor {
	if splitSize <= 0 {
		panic("tensor.Split: splitSize must be > 0")
	}
	ndim := t.Ndim()
	if dim < 0 {
		dim += ndim
	}
	dimLen := t.shape[dim]
	var chunks []*Tensor
	for start := 0; start < dimLen; start += splitSize {
		end := start + splitSize
		if end > dimLen {
			end = dimLen
		}
		chunks = append(chunks, t.Narrow(dim, start, end-start))
	}
	return chunks
}

// Chunk splits t into n equal-sized chunks along dim.
// If the dimension is not divisible by n, the last chunk is smaller.
func Chunk(t *Tensor, n int, dim int) []*Tensor {
	if n <= 0 {
		panic("tensor.Chunk: n must be > 0")
	}
	ndim := t.Ndim()
	if dim < 0 {
		dim += ndim
	}
	dimLen := t.shape[dim]
	size := (dimLen + n - 1) / n // ceiling division
	return Split(t, size, dim)
}
