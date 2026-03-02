package tensor

import "fmt"

// Select returns a slice along dim at index idx, removing that dimension.
// Equivalent to PyTorch t.select(dim, idx) or t[idx] for dim=0.
func (t *Tensor) Select(dim, idx int) *Tensor {
	ndim := t.Ndim()
	if dim < 0 {
		dim += ndim
	}
	if idx < 0 {
		idx += t.shape[dim]
	}
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("tensor.Select: dim %d out of range", dim))
	}
	if idx < 0 || idx >= t.shape[dim] {
		panic(fmt.Sprintf("tensor.Select: index %d out of range [0, %d)", idx, t.shape[dim]))
	}

	// Build output shape (drop dim)
	outShape := make([]int, 0, ndim-1)
	for d := 0; d < ndim; d++ {
		if d != dim {
			outShape = append(outShape, t.shape[d])
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}

	ct := t.ContiguousCopy()
	outSize := totalSize(outShape)
	outData := make([]float64, outSize)

	outerSize := 1
	for d := 0; d < dim; d++ {
		outerSize *= t.shape[d]
	}
	innerSize := 1
	for d := dim + 1; d < ndim; d++ {
		innerSize *= t.shape[d]
	}
	dimLen := t.shape[dim]

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			srcFlat := outer*dimLen*innerSize + idx*innerSize + inner
			dstFlat := outer*innerSize + inner
			outData[dstFlat] = ct.data[srcFlat]
		}
	}
	return New(outData, outShape)
}

// Narrow returns a slice [start, start+length) along dim, keeping the dimension.
// Equivalent to PyTorch t.narrow(dim, start, length).
func (t *Tensor) Narrow(dim, start, length int) *Tensor {
	ndim := t.Ndim()
	if dim < 0 {
		dim += ndim
	}
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("tensor.Narrow: dim %d out of range", dim))
	}
	if start < 0 || length <= 0 || start+length > t.shape[dim] {
		panic(fmt.Sprintf("tensor.Narrow: invalid [%d, %d) for dim size %d", start, start+length, t.shape[dim]))
	}

	outShape := make([]int, ndim)
	copy(outShape, t.shape)
	outShape[dim] = length

	ct := t.ContiguousCopy()
	outData := make([]float64, totalSize(outShape))

	outerSize := 1
	for d := 0; d < dim; d++ {
		outerSize *= t.shape[d]
	}
	innerSize := 1
	for d := dim + 1; d < ndim; d++ {
		innerSize *= t.shape[d]
	}
	dimLen := t.shape[dim]

	for outer := 0; outer < outerSize; outer++ {
		for i := 0; i < length; i++ {
			for inner := 0; inner < innerSize; inner++ {
				srcFlat := outer*dimLen*innerSize + (start+i)*innerSize + inner
				dstFlat := outer*length*innerSize + i*innerSize + inner
				outData[dstFlat] = ct.data[srcFlat]
			}
		}
	}
	return New(outData, outShape)
}

// Index selects rows by integer indices along dim=0.
// Equivalent to PyTorch t[indices].
func (t *Tensor) Index(indices []int) *Tensor {
	if t.Ndim() == 0 {
		panic("tensor.Index: cannot index a scalar")
	}
	rowSize := 1
	for d := 1; d < t.Ndim(); d++ {
		rowSize *= t.shape[d]
	}
	ct := t.ContiguousCopy()

	outShape := make([]int, t.Ndim())
	copy(outShape, t.shape)
	outShape[0] = len(indices)
	outData := make([]float64, len(indices)*rowSize)

	for i, idx := range indices {
		if idx < 0 {
			idx += t.shape[0]
		}
		if idx < 0 || idx >= t.shape[0] {
			panic(fmt.Sprintf("tensor.Index: index %d out of range [0, %d)", idx, t.shape[0]))
		}
		copy(outData[i*rowSize:], ct.data[idx*rowSize:(idx+1)*rowSize])
	}
	return New(outData, outShape)
}

// MaskedSelect returns a 1-D tensor of elements where mask is non-zero.
// Equivalent to PyTorch t.masked_select(mask).
func (t *Tensor) MaskedSelect(mask *Tensor) *Tensor {
	if t.Size() != mask.Size() {
		panic(fmt.Sprintf("tensor.MaskedSelect: size mismatch: t=%d mask=%d", t.Size(), mask.Size()))
	}
	ct := t.ContiguousCopy()
	cm := mask.ContiguousCopy()
	var out []float64
	for i := 0; i < t.Size(); i++ {
		if cm.data[i] != 0 {
			out = append(out, ct.data[i])
		}
	}
	if len(out) == 0 {
		return Zeros(0)
	}
	return New(out, []int{len(out)})
}

// Clamp restricts each element to [min, max].
// Equivalent to PyTorch t.clamp(min, max).
func Clamp(t *Tensor, min, max float64) *Tensor {
	return unary(t, func(x float64) float64 {
		if x < min {
			return min
		}
		if x > max {
			return max
		}
		return x
	})
}

// Where returns a tensor with elements from a where condition != 0, else from b.
// Equivalent to PyTorch torch.where(condition, a, b).
func Where(condition, a, b *Tensor) *Tensor {
	if condition.Size() != a.Size() || a.Size() != b.Size() {
		panic("tensor.Where: all tensors must have the same size")
	}
	cc := condition.ContiguousCopy()
	ca := a.ContiguousCopy()
	cb := b.ContiguousCopy()
	out := make([]float64, a.Size())
	for i := range out {
		if cc.data[i] != 0 {
			out[i] = ca.data[i]
		} else {
			out[i] = cb.data[i]
		}
	}
	return New(out, a.Shape())
}
