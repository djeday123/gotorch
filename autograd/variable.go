package autograd

import "github.com/djeday123/gotorch/tensor"

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

// NewResult creates an intermediate (non-leaf) Variable produced by an op.
// Exported so that the nn package can build custom differentiable layers.
func NewResult(t *tensor.Tensor, gradFn GradFn, children ...*Variable) *Variable {
	return newResult(t, gradFn, children...)
}

// newResult creates an intermediate (non-leaf) Variable produced by an op.
// When gradient computation is disabled (NoGrad context), the result is a
// plain leaf Variable with no grad_fn — identical to calling Detach().
func newResult(t *tensor.Tensor, gradFn GradFn, children ...*Variable) *Variable {
	if !IsGradEnabled() {
		return &Variable{Data: t, isLeaf: true}
	}
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

// ZeroGrad sets this variable's gradient to nil (clears accumulated gradient).
func (v *Variable) ZeroGrad() {
	v.Grad = nil
}

// Detach returns a new leaf Variable sharing the same data but without grad tracking.
func (v *Variable) Detach() *Variable {
	return NewVar(v.Data, false)
}
