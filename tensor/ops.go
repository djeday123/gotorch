package tensor

import "math"

// elementWise applies a binary function element-wise with broadcasting.
// If both inputs are Float32, the result is Float32; otherwise Float64.
func elementWise(a, b *Tensor, fn func(x, y float64) float64) *Tensor {
	isF32 := a.dtype == Float32 && b.dtype == Float32
	if isF32 {
		a, b = a.Float64(), b.Float64()
	}
	ba, bb := broadcast(a, b)
	out := Zeros(ba.shape...)
	ita, itb := newIterator(ba), newIterator(bb)
	for i := 0; ita.hasNext(); i++ {
		out.data[i] = fn(ba.data[ita.next()], bb.data[itb.next()])
	}
	if isF32 {
		return out.Float32()
	}
	return out
}

// unary applies a function to every element.
// Preserves Float32 dtype.
func unary(t *Tensor, fn func(x float64) float64) *Tensor {
	isF32 := t.dtype == Float32
	if isF32 {
		t = t.Float64()
	}
	out := Zeros(t.shape...)
	it := newIterator(t)
	for i := 0; it.hasNext(); i++ {
		out.data[i] = fn(t.data[it.next()])
	}
	if isF32 {
		return out.Float32()
	}
	return out
}

// --- Binary element-wise ---

func Add(a, b *Tensor) *Tensor { return elementWise(a, b, func(x, y float64) float64 { return x + y }) }
func Sub(a, b *Tensor) *Tensor { return elementWise(a, b, func(x, y float64) float64 { return x - y }) }
func Mul(a, b *Tensor) *Tensor { return elementWise(a, b, func(x, y float64) float64 { return x * y }) }
func Div(a, b *Tensor) *Tensor { return elementWise(a, b, func(x, y float64) float64 { return x / y }) }

// --- Scalar ops ---

func AddScalar(t *Tensor, s float64) *Tensor { return unary(t, func(x float64) float64 { return x + s }) }
func SubScalar(t *Tensor, s float64) *Tensor { return unary(t, func(x float64) float64 { return x - s }) }
func MulScalar(t *Tensor, s float64) *Tensor { return unary(t, func(x float64) float64 { return x * s }) }
func DivScalar(t *Tensor, s float64) *Tensor { return unary(t, func(x float64) float64 { return x / s }) }
func PowScalar(t *Tensor, p float64) *Tensor {
	return unary(t, func(x float64) float64 { return math.Pow(x, p) })
}

// --- Unary ---

func Neg(t *Tensor) *Tensor  { return unary(t, func(x float64) float64 { return -x }) }
func Abs(t *Tensor) *Tensor  { return unary(t, func(x float64) float64 { return math.Abs(x) }) }
func Exp(t *Tensor) *Tensor  { return unary(t, func(x float64) float64 { return math.Exp(x) }) }
func Log(t *Tensor) *Tensor  { return unary(t, func(x float64) float64 { return math.Log(x) }) }
func Sqrt(t *Tensor) *Tensor { return unary(t, func(x float64) float64 { return math.Sqrt(x) }) }

// --- Activations ---

func ReLU(t *Tensor) *Tensor {
	return unary(t, func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	})
}

func Sigmoid(t *Tensor) *Tensor {
	return unary(t, func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	})
}

func Tanh(t *Tensor) *Tensor {
	return unary(t, func(x float64) float64 { return math.Tanh(x) })
}

// --- In-place ops ---
//
// These mutate the receiver tensor t and return it. They allocate nothing
// (the whole point) — useful in optimizer steps and gradient accumulation
// where the temporary tensors produced by Add()/Mul()/MulScalar() add up
// to significant GC pressure in long training loops.
//
// Constraints:
//   - Same shape required (no broadcasting): we operate element-wise over
//     the underlying contiguous storage.
//   - Both tensors must have the same dtype.
//   - The receiver MUST be contiguous (panics otherwise) so we can iterate
//     the storage directly. Non-contiguous (views/transposes) need to be
//     materialised first via ContiguousCopy().
//
// To stay safe in autograd-tracked code paths the caller is responsible for
// detaching gradients — modifying a Variable's underlying tensor while it
// is still referenced by a computation graph will corrupt backward.

func (t *Tensor) checkInPlaceSameShape(o *Tensor) {
	if !t.isContiguous() {
		panic("tensor: in-place op requires contiguous receiver — call ContiguousCopy() first")
	}
	if t.dtype != o.dtype {
		panic("tensor: in-place op requires matching dtypes")
	}
	if len(t.shape) != len(o.shape) {
		panic("tensor: in-place op requires matching shapes")
	}
	for i := range t.shape {
		if t.shape[i] != o.shape[i] {
			panic("tensor: in-place op requires matching shapes")
		}
	}
}

// AddInPlace adds o to t element-wise, returning t.
func (t *Tensor) AddInPlace(o *Tensor) *Tensor {
	t.checkInPlaceSameShape(o)
	if t.dtype == Float32 {
		oc := o
		if !o.isContiguous() {
			oc = o.ContiguousCopy()
		}
		for i := range t.f32 {
			t.f32[i] += oc.f32[i]
		}
		return t
	}
	oc := o
	if !o.isContiguous() {
		oc = o.ContiguousCopy()
	}
	for i := range t.data {
		t.data[i] += oc.data[i]
	}
	return t
}

// SubInPlace subtracts o from t element-wise, returning t.
func (t *Tensor) SubInPlace(o *Tensor) *Tensor {
	t.checkInPlaceSameShape(o)
	if t.dtype == Float32 {
		oc := o
		if !o.isContiguous() {
			oc = o.ContiguousCopy()
		}
		for i := range t.f32 {
			t.f32[i] -= oc.f32[i]
		}
		return t
	}
	oc := o
	if !o.isContiguous() {
		oc = o.ContiguousCopy()
	}
	for i := range t.data {
		t.data[i] -= oc.data[i]
	}
	return t
}

// MulInPlace multiplies t by o element-wise, returning t.
func (t *Tensor) MulInPlace(o *Tensor) *Tensor {
	t.checkInPlaceSameShape(o)
	if t.dtype == Float32 {
		oc := o
		if !o.isContiguous() {
			oc = o.ContiguousCopy()
		}
		for i := range t.f32 {
			t.f32[i] *= oc.f32[i]
		}
		return t
	}
	oc := o
	if !o.isContiguous() {
		oc = o.ContiguousCopy()
	}
	for i := range t.data {
		t.data[i] *= oc.data[i]
	}
	return t
}

// AddScalarInPlace adds s to every element of t, returning t.
func (t *Tensor) AddScalarInPlace(s float64) *Tensor {
	if !t.isContiguous() {
		panic("tensor: in-place op requires contiguous receiver")
	}
	if t.dtype == Float32 {
		sf := float32(s)
		for i := range t.f32 {
			t.f32[i] += sf
		}
		return t
	}
	for i := range t.data {
		t.data[i] += s
	}
	return t
}

// MulScalarInPlace multiplies every element of t by s, returning t.
func (t *Tensor) MulScalarInPlace(s float64) *Tensor {
	if !t.isContiguous() {
		panic("tensor: in-place op requires contiguous receiver")
	}
	if t.dtype == Float32 {
		sf := float32(s)
		for i := range t.f32 {
			t.f32[i] *= sf
		}
		return t
	}
	for i := range t.data {
		t.data[i] *= s
	}
	return t
}
