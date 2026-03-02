package nn

import "github.com/djeday123/gotorch/autograd"

// Module is the base interface for all neural network layers.
type Module interface {
	Forward(x *autograd.Variable) *autograd.Variable
	Parameters() []*autograd.Variable
	ZeroGrad()
}

// ZeroGradAll zeros gradients of all parameters in a module.
func ZeroGradAll(m Module) {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}
