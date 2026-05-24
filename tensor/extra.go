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
//
// Uses Welford's online algorithm — single-pass and numerically stable for
// large or shifted values where the textbook E[x²] − E[x]² catastrophically
// cancels. Naïve `(x − mean)²` two-pass works fine for typical activations
// but loses precision when |x| ≫ std(x).
func Var(t *Tensor, dim int, keepdim bool, unbiased bool) *Tensor {
	if dim == -999 {
		// Reduce all elements: one Welford accumulator over the whole tensor.
		mean, m2 := 0.0, 0.0
		count := 0
		it := newIterator(t)
		for it.hasNext() {
			x := t.data[it.next()]
			count++
			delta := x - mean
			mean += delta / float64(count)
			m2 += delta * (x - mean)
		}
		n := float64(count)
		if unbiased && count > 1 {
			n--
		}
		out := Scalar(m2 / n)
		if keepdim {
			ones := make([]int, len(t.shape))
			for i := range ones {
				ones[i] = 1
			}
			return out.Reshape(ones...)
		}
		return out
	}

	// Per-axis Welford: one accumulator per output position.
	mean := Mean(t, dim, true) // [..., 1, ...]
	diff := Sub(t, mean)
	sq := Mul(diff, diff)
	s := Sum(sq, dim, keepdim)

	ndim := len(t.shape)
	d := dim
	if d < 0 {
		d = ndim + d
	}
	n := float64(t.shape[d])
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
//
// O(n log k) via a min-heap of size k. The heap holds the k largest values
// seen so far; whenever the next element is bigger than the heap minimum we
// replace it. At the end we drain the heap in descending order.
//
// The previous O(n·k) selection became painful for large tensors with
// moderate k (e.g. k=100 over 1M elements = 10⁸ ops).
func TopK(t *Tensor, k int) (*Tensor, *Tensor) {
	flat := t.ContiguousCopy()
	n := len(flat.data)
	if k > n {
		k = n
	}
	if k == 0 {
		return New(nil, []int{0}), New(nil, []int{0})
	}

	// Min-heap of {value, originalIndex}, capped at size k.
	h := &topKHeap{}
	for i := 0; i < n; i++ {
		v := flat.data[i]
		if h.Len() < k {
			h.push(topKItem{val: v, idx: i})
		} else if v > h.items[0].val {
			h.items[0] = topKItem{val: v, idx: i}
			h.siftDown(0)
		}
	}

	// Drain in descending order: repeatedly pop the min and stack from the back.
	vals := make([]float64, k)
	idxs := make([]float64, k)
	for i := k - 1; i >= 0; i-- {
		top := h.pop()
		vals[i] = top.val
		idxs[i] = float64(top.idx)
	}
	return New(vals, []int{k}), New(idxs, []int{k})
}

// --- topKHeap: min-heap of (value, originalIndex) used by TopK. ---

type topKItem struct {
	val float64
	idx int
}

type topKHeap struct {
	items []topKItem
}

func (h *topKHeap) Len() int { return len(h.items) }

func (h *topKHeap) push(it topKItem) {
	h.items = append(h.items, it)
	h.siftUp(len(h.items) - 1)
}

func (h *topKHeap) pop() topKItem {
	root := h.items[0]
	last := len(h.items) - 1
	h.items[0] = h.items[last]
	h.items = h.items[:last]
	if last > 0 {
		h.siftDown(0)
	}
	return root
}

func (h *topKHeap) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.items[i].val < h.items[parent].val {
			h.items[i], h.items[parent] = h.items[parent], h.items[i]
			i = parent
		} else {
			break
		}
	}
}

func (h *topKHeap) siftDown(i int) {
	n := len(h.items)
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i
		if left < n && h.items[left].val < h.items[smallest].val {
			smallest = left
		}
		if right < n && h.items[right].val < h.items[smallest].val {
			smallest = right
		}
		if smallest == i {
			return
		}
		h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
		i = smallest
	}
}
