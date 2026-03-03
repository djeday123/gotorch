package tensor

import (
	"fmt"
	"math"
)

// ── Additional creation ops ──────────────────────────────────────────────────

// Full creates a tensor filled with the given value.
func Full(val float64, shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.data {
		t.data[i] = val
	}
	return t
}

// Linspace returns n evenly spaced values from start to end (inclusive).
func Linspace(start, end float64, n int) *Tensor {
	if n <= 0 {
		panic("tensor: Linspace n must be > 0")
	}
	data := make([]float64, n)
	if n == 1 {
		data[0] = start
		return New(data, []int{n})
	}
	step := (end - start) / float64(n-1)
	for i := range data {
		data[i] = start + float64(i)*step
	}
	return New(data, []int{n})
}

// ── Additional elementwise ops ───────────────────────────────────────────────

// Floor returns elementwise floor(x).
func Floor(t *Tensor) *Tensor {
	out := t.ContiguousCopy()
	for i, v := range out.data {
		out.data[i] = math.Floor(v)
	}
	return out
}

// Ceil returns elementwise ceil(x).
func Ceil(t *Tensor) *Tensor {
	out := t.ContiguousCopy()
	for i, v := range out.data {
		out.data[i] = math.Ceil(v)
	}
	return out
}

// Round returns elementwise round(x) (round half to even).
func Round(t *Tensor) *Tensor {
	out := t.ContiguousCopy()
	for i, v := range out.data {
		out.data[i] = math.Round(v)
	}
	return out
}

// Sign returns elementwise sign(x): -1, 0, or 1.
func Sign(t *Tensor) *Tensor {
	out := t.ContiguousCopy()
	for i, v := range out.data {
		if v > 0 {
			out.data[i] = 1
		} else if v < 0 {
			out.data[i] = -1
		} else {
			out.data[i] = 0
		}
	}
	return out
}

// ── Additional reductions ────────────────────────────────────────────────────

// Prod returns the product of all elements (dim=-999) or along dim.
func Prod(t *Tensor, dim int, keepdim bool) *Tensor {
	if dim == -999 {
		return reduceAll(t, 1, func(acc, x float64) float64 { return acc * x })
	}
	return reduceAxis(t, dim, keepdim, 1, func(acc, x float64) float64 { return acc * x })
}

// Var computes variance along dim (unbiased=true uses N-1).
// dim=-999 means all elements.
func Var(t *Tensor, dim int, keepdim bool, unbiased bool) *Tensor {
	mean := Mean(t, dim, true)
	diff := Sub(t, mean)
	sq := Mul(diff, diff)
	s := Sum(sq, dim, keepdim)

	var n float64
	if dim == -999 {
		n = float64(t.Size())
	} else {
		ndim := len(t.shape)
		d := dim
		if d < 0 {
			d = ndim + d
		}
		n = float64(t.shape[d])
	}
	if unbiased && n > 1 {
		n--
	}
	for i := range s.data {
		s.data[i] /= n
	}
	return s
}

// Std computes standard deviation along dim.
func Std(t *Tensor, dim int, keepdim bool, unbiased bool) *Tensor {
	v := Var(t, dim, keepdim, unbiased)
	out := v.ContiguousCopy()
	for i, val := range out.data {
		out.data[i] = math.Sqrt(val)
	}
	return out
}

// Norm computes the p-norm of the tensor (all elements).
// p=2 → L2, p=1 → L1, p=0 → count of non-zeros.
func Norm(t *Tensor, p float64) *Tensor {
	flat := t.ContiguousCopy()
	if p == 0 {
		count := 0.0
		for _, v := range flat.data {
			if v != 0 {
				count++
			}
		}
		return Scalar(count)
	}
	if p == math.Inf(1) {
		maxV := 0.0
		for _, v := range flat.data {
			if av := math.Abs(v); av > maxV {
				maxV = av
			}
		}
		return Scalar(maxV)
	}
	sum := 0.0
	for _, v := range flat.data {
		sum += math.Pow(math.Abs(v), p)
	}
	return Scalar(math.Pow(sum, 1.0/p))
}

// NormDim computes the p-norm along a specific dimension.
func NormDim(t *Tensor, p float64, dim int, keepdim bool) *Tensor {
	ndim := len(t.shape)
	if dim < 0 {
		dim = ndim + dim
	}
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("tensor: NormDim dim %d out of range", dim))
	}

	outShape := make([]int, 0, ndim)
	for i, d := range t.shape {
		if i == dim {
			if keepdim {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}

	out := Zeros(outShape...)
	tc := t.ContiguousCopy()
	innerSize := 1
	for i := dim + 1; i < ndim; i++ {
		innerSize *= t.shape[i]
	}
	outerSize := 1
	for i := 0; i < dim; i++ {
		outerSize *= t.shape[i]
	}
	dimSize := t.shape[dim]

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			outIdx := outer*innerSize + inner
			sum := 0.0
			for d := 0; d < dimSize; d++ {
				v := tc.data[outer*dimSize*innerSize+d*innerSize+inner]
				sum += math.Pow(math.Abs(v), p)
			}
			out.data[outIdx] = math.Pow(sum, 1.0/p)
		}
	}
	return out
}

// Cumsum computes the cumulative sum along dim.
func Cumsum(t *Tensor, dim int) *Tensor {
	ndim := len(t.shape)
	if dim < 0 {
		dim = ndim + dim
	}
	tc := t.ContiguousCopy()
	out := Zeros(tc.shape...)

	innerSize := 1
	for i := dim + 1; i < ndim; i++ {
		innerSize *= tc.shape[i]
	}
	outerSize := 1
	for i := 0; i < dim; i++ {
		outerSize *= tc.shape[i]
	}
	dimSize := tc.shape[dim]

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			acc := 0.0
			for d := 0; d < dimSize; d++ {
				idx := outer*dimSize*innerSize + d*innerSize + inner
				acc += tc.data[idx]
				out.data[idx] = acc
			}
		}
	}
	return out
}

// TopK returns the top-k values and their indices along dim=last.
// Returns (values, indices).
func TopK(t *Tensor, k int) (*Tensor, *Tensor) {
	flat := t.ContiguousCopy()
	n := len(flat.data)
	if k > n {
		k = n
	}
	// Simple selection — O(n*k), fine for small k
	used := make([]bool, n)
	vals := make([]float64, k)
	idxs := make([]float64, k)
	for i := 0; i < k; i++ {
		bestVal := math.Inf(-1)
		bestIdx := -1
		for j, v := range flat.data {
			if !used[j] && v > bestVal {
				bestVal = v
				bestIdx = j
			}
		}
		vals[i] = bestVal
		idxs[i] = float64(bestIdx)
		used[bestIdx] = true
	}
	return New(vals, []int{k}), New(idxs, []int{k})
}
