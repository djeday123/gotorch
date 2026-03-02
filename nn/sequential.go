package nn

import "github.com/djeday123/gotorch/autograd"

// Sequential chains multiple Modules, passing the output of each as input to the next.
type Sequential struct {
	layers []Module
}

// NewSequential creates a Sequential with the given layers (in order).
func NewSequential(layers ...Module) *Sequential {
	return &Sequential{layers: layers}
}

// Forward passes x through each layer in order.
func (s *Sequential) Forward(x *autograd.Variable) *autograd.Variable {
	out := x
	for _, l := range s.layers {
		out = l.Forward(out)
	}
	return out
}

// Parameters collects parameters from all child modules.
func (s *Sequential) Parameters() []*autograd.Variable {
	var params []*autograd.Variable
	for _, l := range s.layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

// ZeroGrad zeros gradients of all child modules.
func (s *Sequential) ZeroGrad() {
	for _, l := range s.layers {
		l.ZeroGrad()
	}
}
