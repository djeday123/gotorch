package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// DType represents the element type of a tensor.
type DType int

const (
	Float64 DType = iota // default
	Float32
)

// Tensor is an N-dimensional array.
type Tensor struct {
	data    []float64
	f32     []float32 // populated when dtype == Float32
	dtype   DType
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

// DType returns the element type of the tensor.
func (t *Tensor) DType() DType { return t.dtype }

// NewF32 creates a float32 Tensor from existing data and shape.
func NewF32(data []float32, shape []int) *Tensor {
	if len(data) != totalSize(shape) {
		panic(fmt.Sprintf("tensor: data length %d does not match shape %v", len(data), shape))
	}
	s := make([]int, len(shape))
	copy(s, shape)
	d := make([]float32, len(data))
	copy(d, data)
	return &Tensor{f32: d, dtype: Float32, shape: s, strides: computeStrides(s)}
}

// ZerosF32 creates a float32 zero tensor.
func ZerosF32(shape ...int) *Tensor {
	s := make([]int, len(shape))
	copy(s, shape)
	return &Tensor{f32: make([]float32, totalSize(s)), dtype: Float32, shape: s, strides: computeStrides(s)}
}

// OnesF32 creates a float32 ones tensor.
func OnesF32(shape ...int) *Tensor {
	t := ZerosF32(shape...)
	for i := range t.f32 {
		t.f32[i] = 1
	}
	return t
}

// RandF32 creates a float32 tensor with uniform random values in [0,1).
func RandF32(shape ...int) *Tensor {
	t := ZerosF32(shape...)
	for i := range t.f32 {
		t.f32[i] = float32(rand.Float64())
	}
	return t
}

// RandNF32 creates a float32 tensor with standard normal random values.
func RandNF32(shape ...int) *Tensor {
	t := ZerosF32(shape...)
	for i := range t.f32 {
		t.f32[i] = float32(rand.NormFloat64())
	}
	return t
}

// Float32 casts this tensor to Float32 dtype (copy).
func (t *Tensor) Float32() *Tensor {
	if t.dtype == Float32 {
		return t
	}
	ct := t.ContiguousCopy()
	f32 := make([]float32, len(ct.data))
	for i, v := range ct.data {
		f32[i] = float32(v)
	}
	return &Tensor{f32: f32, dtype: Float32, shape: ct.shape, strides: ct.strides}
}

// Float64 casts this tensor to Float64 dtype (copy).
func (t *Tensor) Float64() *Tensor {
	if t.dtype == Float64 {
		return t
	}
	d := make([]float64, len(t.f32))
	for i, v := range t.f32 {
		d[i] = float64(v)
	}
	s := make([]int, len(t.shape))
	copy(s, t.shape)
	return &Tensor{data: d, dtype: Float64, shape: s, strides: computeStrides(s)}
}

// DataF32 returns a flat copy of the data as []float32.
// Panics if dtype is not Float32.
func (t *Tensor) DataF32() []float32 {
	if t.dtype != Float32 {
		panic("tensor.DataF32: tensor is not Float32")
	}
	out := make([]float32, len(t.f32))
	copy(out, t.f32)
	return out
}

// f64view returns the float64 backing array, converting from f32 if needed.
// Used internally by ops that always work in float64.
func (t *Tensor) f64view() []float64 {
	if t.dtype == Float64 {
		return t.data
	}
	d := make([]float64, len(t.f32))
	for i, v := range t.f32 {
		d[i] = float64(v)
	}
	return d
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
	if t.dtype == Float32 {
		return float64(t.f32[t.offset])
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
	idx := t.flatIndex(indices)
	if t.dtype == Float32 {
		return float64(t.f32[idx])
	}
	return t.data[idx]
}

// Set sets the element at the given multi-dim indices.
func (t *Tensor) Set(val float64, indices ...int) {
	idx := t.flatIndex(indices)
	if t.dtype == Float32 {
		t.f32[idx] = float32(val)
		return
	}
	t.data[idx] = val
}

// Data returns a flat copy of the underlying data in logical order (as float64).
func (t *Tensor) Data() []float64 {
	ct := t.ContiguousCopy()
	if ct.dtype == Float32 {
		out := make([]float64, len(ct.f32))
		for i, v := range ct.f32 {
			out[i] = float64(v)
		}
		return out
	}
	return ct.data
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
