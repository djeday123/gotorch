// Package data provides DataLoader and Dataset primitives, similar to
// torch.utils.data in PyTorch.
package data

import (
	"fmt"

	"github.com/djeday123/gotorch/tensor"
)

// Dataset is the interface that all datasets must implement.
type Dataset interface {
	// Len returns the total number of samples.
	Len() int
	// Get returns the i-th sample as (input, target).
	// Both tensors may be nil if the dataset is unsupervised.
	Get(idx int) (*tensor.Tensor, *tensor.Tensor)
}

// ── TensorDataset ─────────────────────────────────────────────────────────────

// TensorDataset wraps two tensors (X, Y) as a dataset, slicing along dim 0.
type TensorDataset struct {
	X *tensor.Tensor // [N, ...] inputs
	Y *tensor.Tensor // [N, ...] targets (may be nil)
	n int
}

// NewTensorDataset creates a TensorDataset.
// X must have at least one dimension; Y may be nil (unsupervised).
func NewTensorDataset(X, Y *tensor.Tensor) *TensorDataset {
	n := X.Shape()[0]
	if Y != nil && Y.Shape()[0] != n {
		panic(fmt.Sprintf("data: X has %d samples but Y has %d", n, Y.Shape()[0]))
	}
	return &TensorDataset{X: X, Y: Y, n: n}
}

func (d *TensorDataset) Len() int { return d.n }

// Get returns the i-th row of X and Y (if Y is not nil).
func (d *TensorDataset) Get(idx int) (*tensor.Tensor, *tensor.Tensor) {
	if idx < 0 || idx >= d.n {
		panic(fmt.Sprintf("data: index %d out of range [0, %d)", idx, d.n))
	}
	xi := rowSlice(d.X, idx)
	if d.Y == nil {
		return xi, nil
	}
	return xi, rowSlice(d.Y, idx)
}

// rowSlice extracts row idx from a tensor of shape [N, ...].
// Returns a tensor of shape [...] (the remaining dims).
func rowSlice(t *tensor.Tensor, idx int) *tensor.Tensor {
	shape := t.Shape()
	rowSize := 1
	for _, d := range shape[1:] {
		rowSize *= d
	}
	flat := t.ContiguousCopy().Data()
	rowData := make([]float64, rowSize)
	copy(rowData, flat[idx*rowSize:(idx+1)*rowSize])

	if len(shape) == 1 {
		// scalar
		return tensor.Scalar(rowData[0])
	}
	return tensor.New(rowData, shape[1:])
}
