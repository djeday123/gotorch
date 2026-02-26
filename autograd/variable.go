package autograd

import (
	"gotorch_v1/tensor"
)

// GradFn is implemented by every differentiable operation.
// Apply receives the upstream gradient and returns gradients
// for each of its inputs (in the same order as children).
type GradFn interface {
	Apply(grad *tensor.Tensor) []*tensor.Tensor
}

// Variable wraps a Tensor and tracks the computation graph.
type Variable struct {
	Data         *tensor.Tensor
	Grad         *tensor.Tensor
	RequiresGrad bool
	isLeaf       bool
	gradFn       GradFn
	children     []*Variable
}

// NewVar creates a leaf Variable.
func NewVar(t *tensor.Tensor, requiresGrad bool) *Variable {
	return &Variable{
		Data:         t,
		RequiresGrad: requiresGrad,
		isLeaf:       true,
	}
}

// newResult creates an intermediate (non-leaf) Variable produced by an op.
func newResult(t *tensor.Tensor, gradFn GradFn, children ...*Variable) *Variable {
	needsGrad := false
	for _, c := range children {
		if c.RequiresGrad {
			needsGrad = true
			break
		}
	}
	v := &Variable{
		Data:         t,
		RequiresGrad: needsGrad,
		isLeaf:       false,
		gradFn:       gradFn,
		children:     children,
	}
	return v
}

// Backward runs backpropagation from this variable (must be scalar).
func (v *Variable) Backward() {
	if v.Data.Size() != 1 {
		panic("autograd: Backward() must be called on a scalar variable")
	}
	grad := tensor.Ones(v.Data.Shape()...)
	backward(v, grad)
}

// BackwardWithGrad runs backpropagation with a custom upstream gradient.
func (v *Variable) BackwardWithGrad(grad *tensor.Tensor) {
	backward(v, grad)
}

// ZeroGrad sets this variable's gradient to zero.
func (v *Variable) ZeroGrad() {
	if v.Grad != nil {
		v.Grad = tensor.Zeros(v.Data.Shape()...)
	}
}

// Detach returns a new leaf Variable sharing the same data but without grad tracking.
func (v *Variable) Detach() *Variable {
	return NewVar(v.Data, false)
}
