package tensor

import (
	"fmt"
	"math"
)

// reduceAxis reduces along a single axis using the given function.
// fn receives (accumulator, next_value) and returns new accumulator.
func reduceAxis(t *Tensor, dim int, keepdim bool, init float64, fn func(acc, x float64) float64) *Tensor {
	ndim := len(t.shape)
	if dim < 0 {
		dim = ndim + dim
	}
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("tensor: reduce dim %d out of range for %dD tensor", dim, ndim))
	}

	// Output shape
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
	for i := range out.data {
		out.data[i] = init
	}

	// Iterate over all elements of t
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
			for d := 0; d < dimSize; d++ {
				srcIdx := outer*dimSize*innerSize + d*innerSize + inner
				out.data[outIdx] = fn(out.data[outIdx], tc.data[srcIdx])
			}
		}
	}
	return out
}

// reduceAll reduces all elements.
func reduceAll(t *Tensor, init float64, fn func(acc, x float64) float64) *Tensor {
	flat := t.ContiguousCopy()
	acc := init
	for _, v := range flat.data {
		acc = fn(acc, v)
	}
	return Scalar(acc)
}

// Sum reduces along dim. dim=-999 means all elements.
func Sum(t *Tensor, dim int, keepdim bool) *Tensor {
	if dim == -999 {
		return reduceAll(t, 0, func(acc, x float64) float64 { return acc + x })
	}
	return reduceAxis(t, dim, keepdim, 0, func(acc, x float64) float64 { return acc + x })
}

// Mean reduces along dim. dim=-999 means all elements.
func Mean(t *Tensor, dim int, keepdim bool) *Tensor {
	if dim == -999 {
		s := reduceAll(t, 0, func(acc, x float64) float64 { return acc + x })
		s.data[0] /= float64(t.Size())
		return s
	}
	s := reduceAxis(t, dim, keepdim, 0, func(acc, x float64) float64 { return acc + x })
	n := float64(t.shape[dim])
	for i := range s.data {
		s.data[i] /= n
	}
	return s
}

// Max reduces along dim. dim=-999 means all elements.
func Max(t *Tensor, dim int, keepdim bool) *Tensor {
	if dim == -999 {
		return reduceAll(t, math.Inf(-1), func(acc, x float64) float64 {
			if x > acc {
				return x
			}
			return acc
		})
	}
	return reduceAxis(t, dim, keepdim, math.Inf(-1), func(acc, x float64) float64 {
		if x > acc {
			return x
		}
		return acc
	})
}

// Min reduces along dim. dim=-999 means all elements.
func Min(t *Tensor, dim int, keepdim bool) *Tensor {
	if dim == -999 {
		return reduceAll(t, math.Inf(1), func(acc, x float64) float64 {
			if x < acc {
				return x
			}
			return acc
		})
	}
	return reduceAxis(t, dim, keepdim, math.Inf(1), func(acc, x float64) float64 {
		if x < acc {
			return x
		}
		return acc
	})
}

// ArgMax returns indices of maximum values along dim.
func ArgMax(t *Tensor, dim int) *Tensor {
	ndim := len(t.shape)
	if dim < 0 {
		dim = ndim + dim
	}
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

	outShape := make([]int, 0, ndim-1)
	for i, d := range t.shape {
		if i != dim {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	out := Zeros(outShape...)

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			bestIdx := 0
			bestVal := math.Inf(-1)
			for d := 0; d < dimSize; d++ {
				v := tc.data[outer*dimSize*innerSize+d*innerSize+inner]
				if v > bestVal {
					bestVal = v
					bestIdx = d
				}
			}
			out.data[outer*innerSize+inner] = float64(bestIdx)
		}
	}
	return out
}

// Softmax computes numerically stable softmax along dim.
func Softmax(t *Tensor, dim int) *Tensor {
	// max subtraction for numerical stability
	maxVals := Max(t, dim, true)
	shifted := Sub(t, maxVals)
	exps := Exp(shifted)
	sumExps := Sum(exps, dim, true)
	return Div(exps, sumExps)
}

// LogSoftmax computes log(softmax(x)) in a numerically stable way.
func LogSoftmax(t *Tensor, dim int) *Tensor {
	maxVals := Max(t, dim, true)
	shifted := Sub(t, maxVals)
	exps := Exp(shifted)
	sumExps := Sum(exps, dim, true)
	logSum := Log(sumExps)
	return Sub(shifted, logSum)
}
