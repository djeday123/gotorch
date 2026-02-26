package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Tensor is an N-dimensional array.
type Tensor struct {
	data    []float64
	shape   []int
	strides []int
	offset  int
}

// computeStrides returns row-major (C-order) strides for a shape.
func computeStrides(shape []int) []int {
	ndim := len(shape)
	strides := make([]int, ndim)
	if ndim == 0 {
		return strides
	}
	strides[ndim-1] = 1
	for i := ndim - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// totalSize returns the product of all dims.
func totalSize(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// New creates a Tensor from existing data and shape.
func New(data []float64, shape []int) *Tensor {
	if len(data) != totalSize(shape) {
		panic(fmt.Sprintf("tensor: data length %d does not match shape %v (size %d)", len(data), shape, totalSize(shape)))
	}
	s := make([]int, len(shape))
	copy(s, shape)
	d := make([]float64, len(data))
	copy(d, data)
	return &Tensor{data: d, shape: s, strides: computeStrides(s)}
}

// Zeros creates a zero-filled tensor.
func Zeros(shape ...int) *Tensor {
	s := make([]int, len(shape))
	copy(s, shape)
	return &Tensor{data: make([]float64, totalSize(s)), shape: s, strides: computeStrides(s)}
}

// Ones creates a ones-filled tensor.
func Ones(shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.data {
		t.data[i] = 1
	}
	return t
}

// Rand creates a tensor filled with uniform random values in [0, 1).
func Rand(shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.data {
		t.data[i] = rand.Float64()
	}
	return t
}

// RandN creates a tensor filled with standard normal random values.
func RandN(shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.data {
		t.data[i] = rand.NormFloat64()
	}
	return t
}

// Arange creates a 1-D tensor [start, stop) with step.
func Arange(start, stop, step float64) *Tensor {
	n := int(math.Ceil((stop - start) / step))
	if n <= 0 {
		n = 0
	}
	data := make([]float64, n)
	for i := range data {
		data[i] = start + float64(i)*step
	}
	return &Tensor{data: data, shape: []int{n}, strides: []int{1}}
}

// Eye creates an n×n identity matrix.
func Eye(n int) *Tensor {
	t := Zeros(n, n)
	for i := 0; i < n; i++ {
		t.data[i*n+i] = 1
	}
	return t
}

// Scalar creates a 0-dimensional scalar tensor.
func Scalar(v float64) *Tensor {
	return &Tensor{data: []float64{v}, shape: []int{}, strides: []int{}}
}

// Shape returns the tensor shape.
func (t *Tensor) Shape() []int {
	s := make([]int, len(t.shape))
	copy(s, t.shape)
	return s
}

// Ndim returns the number of dimensions.
func (t *Tensor) Ndim() int { return len(t.shape) }

// Size returns the total number of elements.
func (t *Tensor) Size() int { return totalSize(t.shape) }

// Item returns the scalar value of a single-element tensor.
func (t *Tensor) Item() float64 {
	if t.Size() != 1 {
		panic("tensor: Item() called on non-scalar tensor")
	}
	return t.data[t.offset]
}

// flatIndex computes the flat index for the given multi-dim indices.
func (t *Tensor) flatIndex(indices []int) int {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("tensor: expected %d indices, got %d", len(t.shape), len(indices)))
	}
	idx := t.offset
	for i, ind := range indices {
		if ind < 0 {
			ind += t.shape[i]
		}
		if ind < 0 || ind >= t.shape[i] {
			panic(fmt.Sprintf("tensor: index %d out of range for dim %d (size %d)", ind, i, t.shape[i]))
		}
		idx += ind * t.strides[i]
	}
	return idx
}

// At returns the element at the given multi-dim indices.
func (t *Tensor) At(indices ...int) float64 {
	return t.data[t.flatIndex(indices)]
}

// Set sets the element at the given multi-dim indices.
func (t *Tensor) Set(val float64, indices ...int) {
	t.data[t.flatIndex(indices)] = val
}

// Data returns a flat copy of the underlying data in logical order.
func (t *Tensor) Data() []float64 {
	return t.ContiguousCopy().data
}

// String prints the tensor in a readable format.
func (t *Tensor) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Tensor(shape=%v)\n", t.shape))
	if len(t.shape) == 0 {
		sb.WriteString(fmt.Sprintf("%.4f", t.data[t.offset]))
		return sb.String()
	}
	t.printRecursive(&sb, 0, make([]int, len(t.shape)))
	return sb.String()
}

func (t *Tensor) printRecursive(sb *strings.Builder, dim int, indices []int) {
	if dim == len(t.shape)-1 {
		sb.WriteString("[")
		for i := 0; i < t.shape[dim]; i++ {
			indices[dim] = i
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%.4f", t.At(indices...)))
		}
		sb.WriteString("]")
		return
	}
	sb.WriteString("[")
	for i := 0; i < t.shape[dim]; i++ {
		indices[dim] = i
		if i > 0 {
			sb.WriteString(",\n" + strings.Repeat(" ", dim+1))
		}
		t.printRecursive(sb, dim+1, indices)
	}
	sb.WriteString("]")
}
