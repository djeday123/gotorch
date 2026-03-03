package nn

import "github.com/djeday123/gotorch/autograd"

// ModuleList holds an ordered list of Modules.
// It implements Module itself, calling Forward sequentially.
type ModuleList struct {
	Modules []Module
}

// NewModuleList creates a ModuleList from the given modules.
func NewModuleList(modules ...Module) *ModuleList {
	ms := make([]Module, len(modules))
	copy(ms, modules)
	return &ModuleList{Modules: ms}
}

// Append adds a module to the end of the list.
func (ml *ModuleList) Append(m Module) {
	ml.Modules = append(ml.Modules, m)
}

// Get returns the module at index i.
func (ml *ModuleList) Get(i int) Module {
	return ml.Modules[i]
}

// Len returns the number of modules.
func (ml *ModuleList) Len() int { return len(ml.Modules) }

// Forward runs the input through all modules sequentially.
func (ml *ModuleList) Forward(x *autograd.Variable) *autograd.Variable {
	out := x
	for _, m := range ml.Modules {
		out = m.Forward(out)
	}
	return out
}

// Parameters collects parameters from all modules (no deduplication).
func (ml *ModuleList) Parameters() []*autograd.Variable {
	var params []*autograd.Variable
	for _, m := range ml.Modules {
		params = append(params, m.Parameters()...)
	}
	return params
}

// ZeroGrad zeros gradients in all modules.
func (ml *ModuleList) ZeroGrad() {
	for _, m := range ml.Modules {
		m.ZeroGrad()
	}
}
