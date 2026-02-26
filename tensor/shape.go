package tensor

import "fmt"

// ContiguousCopy returns a new contiguous tensor with the same logical data.
func (t *Tensor) ContiguousCopy() *Tensor {
	out := Zeros(t.shape...)
	it := newIterator(t)
	for i := 0; it.hasNext(); i++ {
		out.data[i] = t.data[it.next()]
	}
	return out
}

// isContiguous returns true if the tensor data is laid out without gaps.
func (t *Tensor) isContiguous() bool {
	expected := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		if t.shape[i] == 1 {
			continue
		}
		if t.strides[i] != expected {
			return false
		}
		expected *= t.shape[i]
	}
	return true
}

// Reshape returns a reshaped tensor. Returns a view if contiguous, else copies.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	// Handle -1 (infer dim)
	inferred := -1
	size := 1
	for i, d := range shape {
		if d == -1 {
			if inferred >= 0 {
				panic("tensor: only one dimension can be inferred (-1)")
			}
			inferred = i
		} else {
			size *= d
		}
	}
	if inferred >= 0 {
		shape[inferred] = t.Size() / size
		size *= shape[inferred]
	}
	if size != t.Size() {
		panic(fmt.Sprintf("tensor: cannot reshape tensor of size %d into shape %v", t.Size(), shape))
	}

	newShape := make([]int, len(shape))
	copy(newShape, shape)

	if t.isContiguous() {
		// View: share data buffer
		return &Tensor{
			data:    t.data,
			shape:   newShape,
			strides: computeStrides(newShape),
			offset:  t.offset,
		}
	}
	// Need a copy
	c := t.ContiguousCopy()
	c.shape = newShape
	c.strides = computeStrides(newShape)
	return c
}

// Flatten returns a 1-D view (or copy).
func (t *Tensor) Flatten() *Tensor {
	return t.Reshape(t.Size())
}

// Squeeze removes all size-1 dimensions.
func (t *Tensor) Squeeze() *Tensor {
	newShape := make([]int, 0, len(t.shape))
	for _, d := range t.shape {
		if d != 1 {
			newShape = append(newShape, d)
		}
	}
	if len(newShape) == 0 {
		newShape = []int{1}
	}
	return t.Reshape(newShape...)
}

// Unsqueeze inserts a size-1 dimension at position dim.
func (t *Tensor) Unsqueeze(dim int) *Tensor {
	ndim := len(t.shape)
	if dim < 0 {
		dim = ndim + 1 + dim
	}
	if dim < 0 || dim > ndim {
		panic(fmt.Sprintf("tensor: unsqueeze dim %d out of range for %dD tensor", dim, ndim))
	}
	newShape := make([]int, 0, ndim+1)
	newShape = append(newShape, t.shape[:dim]...)
	newShape = append(newShape, 1)
	newShape = append(newShape, t.shape[dim:]...)
	return t.Reshape(newShape...)
}

// Transpose permutes dimensions. If no dims given, reverses all axes.
func (t *Tensor) Transpose(dims ...int) *Tensor {
	ndim := len(t.shape)
	if len(dims) == 0 {
		dims = make([]int, ndim)
		for i := range dims {
			dims[i] = ndim - 1 - i
		}
	}
	if len(dims) != ndim {
		panic(fmt.Sprintf("tensor: transpose expects %d dims, got %d", ndim, len(dims)))
	}
	seen := make(map[int]bool)
	for _, d := range dims {
		if d < 0 || d >= ndim {
			panic(fmt.Sprintf("tensor: transpose dim %d out of range", d))
		}
		if seen[d] {
			panic(fmt.Sprintf("tensor: duplicate dim %d in transpose", d))
		}
		seen[d] = true
	}

	newShape := make([]int, ndim)
	newStrides := make([]int, ndim)
	for i, d := range dims {
		newShape[i] = t.shape[d]
		newStrides[i] = t.strides[d]
	}
	return &Tensor{data: t.data, shape: newShape, strides: newStrides, offset: t.offset}
}

// T is a shorthand for Transpose() on 2-D tensors.
func (t *Tensor) T() *Tensor {
	if len(t.shape) != 2 {
		panic("tensor: T() only works on 2D tensors")
	}
	return t.Transpose(1, 0)
}

// broadcastShapes computes the output shape for broadcasting two shapes.
func broadcastShapes(a, b []int) []int {
	la, lb := len(a), len(b)
	n := la
	if lb > n {
		n = lb
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		ai := 1
		bi := 1
		if ia := i - (n - la); ia >= 0 {
			ai = a[ia]
		}
		if ib := i - (n - lb); ib >= 0 {
			bi = b[ib]
		}
		if ai == bi {
			out[i] = ai
		} else if ai == 1 {
			out[i] = bi
		} else if bi == 1 {
			out[i] = ai
		} else {
			panic(fmt.Sprintf("tensor: shapes %v and %v are not broadcastable", a, b))
		}
	}
	return out
}

// broadcastTo returns a tensor broadcast to the target shape (no data copy; virtual strides).
func broadcastTo(t *Tensor, shape []int) *Tensor {
	ndim := len(shape)
	tndim := len(t.shape)

	newStrides := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		ti := i - (ndim - tndim)
		if ti < 0 {
			newStrides[i] = 0 // virtual axis
		} else if t.shape[ti] == 1 {
			newStrides[i] = 0 // broadcast axis
		} else {
			newStrides[i] = t.strides[ti]
		}
	}
	s := make([]int, ndim)
	copy(s, shape)
	return &Tensor{data: t.data, shape: s, strides: newStrides, offset: t.offset}
}

// broadcast prepares two tensors for element-wise ops by aligning their shapes.
func broadcast(a, b *Tensor) (*Tensor, *Tensor) {
	outShape := broadcastShapes(a.shape, b.shape)
	return broadcastTo(a, outShape), broadcastTo(b, outShape)
}

// iterator for traversing a tensor in logical order.
type iterator struct {
	t       *Tensor
	indices []int
	pos     int
	size    int
}

func newIterator(t *Tensor) *iterator {
	return &iterator{t: t, indices: make([]int, len(t.shape)), size: t.Size()}
}

func (it *iterator) hasNext() bool { return it.pos < it.size }

func (it *iterator) next() int {
	// compute flat index from indices
	idx := it.t.offset
	for i, ind := range it.indices {
		idx += ind * it.t.strides[i]
	}
	// advance indices
	for dim := len(it.indices) - 1; dim >= 0; dim-- {
		it.indices[dim]++
		if it.indices[dim] < it.t.shape[dim] {
			break
		}
		it.indices[dim] = 0
	}
	it.pos++
	return idx
}
