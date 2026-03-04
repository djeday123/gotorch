package nn

import "github.com/djeday123/gotorch/autograd"

// ---------------------------------------------------------------------------
// ModuleDict — ordered map of named modules
// ---------------------------------------------------------------------------

// ModuleDict holds a named, ordered collection of modules.
// Equivalent to torch.nn.ModuleDict.
//
//	d := nn.NewModuleDict(map[string]nn.Module{
//	    "linear": nn.NewLinear(16, 8, true),
//	    "relu":   &nn.ReLUModule{},
//	})
type ModuleDict struct {
	Keys    []string
	modules map[string]Module
}

// NewModuleDict creates a ModuleDict from a map. Key order is not guaranteed
// from the map — use Add to build an ordered dict.
func NewModuleDict(m map[string]Module) *ModuleDict {
	d := &ModuleDict{modules: make(map[string]Module)}
	for k, v := range m {
		d.Keys = append(d.Keys, k)
		d.modules[k] = v
	}
	return d
}

// Add appends a named module (preserving insertion order).
func (d *ModuleDict) Add(key string, m Module) {
	if _, exists := d.modules[key]; !exists {
		d.Keys = append(d.Keys, key)
	}
	if d.modules == nil {
		d.modules = make(map[string]Module)
	}
	d.modules[key] = m
}

// Get retrieves a module by name. Returns nil if not found.
func (d *ModuleDict) Get(key string) Module {
	return d.modules[key]
}

// Len returns the number of modules.
func (d *ModuleDict) Len() int { return len(d.modules) }

// Parameters collects parameters from all modules in insertion order.
func (d *ModuleDict) Parameters() []*autograd.Variable {
	var ps []*autograd.Variable
	for _, k := range d.Keys {
		ps = append(ps, d.modules[k].Parameters()...)
	}
	return ps
}

// ZeroGrad zeros gradients in all modules.
func (d *ModuleDict) ZeroGrad() {
	for _, m := range d.modules {
		m.ZeroGrad()
	}
}

// ---------------------------------------------------------------------------
// StackedLSTM — multi-layer LSTM
// ---------------------------------------------------------------------------

// StackedLSTM stacks numLayers LSTM cells, feeding the hidden output of
// layer i as the input to layer i+1. Equivalent to LSTM(numLayers=N).
//
//	lstm := nn.NewStackedLSTM(inputSize, hiddenSize, numLayers)
//	h, c, out := lstm.Forward(x, nil, nil)  // x: [T, inputSize]
type StackedLSTM struct {
	Layers     []*LSTM
	NumLayers  int
	HiddenSize int
}

// NewStackedLSTM creates a multi-layer LSTM.
func NewStackedLSTM(inputSize, hiddenSize, numLayers int) *StackedLSTM {
	layers := make([]*LSTM, numLayers)
	layers[0] = NewLSTM(inputSize, hiddenSize)
	for i := 1; i < numLayers; i++ {
		layers[i] = NewLSTM(hiddenSize, hiddenSize)
	}
	return &StackedLSTM{Layers: layers, NumLayers: numLayers, HiddenSize: hiddenSize}
}

// LSTMState holds hidden and cell state for one layer.
type StackedLSTMState struct {
	H []*autograd.Variable // [numLayers] each [hiddenSize]
	C []*autograd.Variable // [numLayers] each [hiddenSize]
}

// Forward runs input x [T, inputSize] through all layers.
// Returns (finalH, finalC, allOutputs) where allOutputs is [T, hiddenSize].
// Pass nil states to use zero initialisation.
func (s *StackedLSTM) Forward(x *autograd.Variable, state *StackedLSTMState) (*autograd.Variable, *autograd.Variable, *autograd.Variable) {
	T := x.Data.Shape()[0]

	// Build initial states
	initH := make([]*autograd.Variable, s.NumLayers)
	initC := make([]*autograd.Variable, s.NumLayers)
	for i := range initH {
		if state != nil && i < len(state.H) {
			initH[i] = state.H[i]
			initC[i] = state.C[i]
		} else {
			initH[i] = autograd.NewVar(zerosTensor1D(s.HiddenSize), false)
			initC[i] = autograd.NewVar(zerosTensor1D(s.HiddenSize), false)
		}
	}

	// Run through layers
	curInput := x
	finalH := make([]*autograd.Variable, s.NumLayers)
	finalC := make([]*autograd.Variable, s.NumLayers)

	for l, layer := range s.Layers {
		h, c, out := layer.ForwardSequence(curInput, initH[l], initC[l])
		finalH[l] = h
		finalC[l] = c
		_ = T
		curInput = out
	}

	return finalH[s.NumLayers-1], finalC[s.NumLayers-1], curInput
}

// Parameters returns all parameters from all layers.
func (s *StackedLSTM) Parameters() []*autograd.Variable {
	var ps []*autograd.Variable
	for _, l := range s.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}

// ZeroGrad zeros all gradients.
func (s *StackedLSTM) ZeroGrad() {
	for _, l := range s.Layers {
		l.ZeroGrad()
	}
}

// StackedGRU — multi-layer GRU
type StackedGRU struct {
	Layers     []*GRU
	NumLayers  int
	HiddenSize int
}

// NewStackedGRU creates a multi-layer GRU.
func NewStackedGRU(inputSize, hiddenSize, numLayers int) *StackedGRU {
	layers := make([]*GRU, numLayers)
	layers[0] = NewGRU(inputSize, hiddenSize)
	for i := 1; i < numLayers; i++ {
		layers[i] = NewGRU(hiddenSize, hiddenSize)
	}
	return &StackedGRU{Layers: layers, NumLayers: numLayers, HiddenSize: hiddenSize}
}

// Forward runs x [T, inputSize] through all GRU layers.
// Returns (finalH, allOutputs [T, hiddenSize]).
func (s *StackedGRU) Forward(x *autograd.Variable, initH *autograd.Variable) (*autograd.Variable, *autograd.Variable) {
	curInput := x
	var lastH *autograd.Variable

	for _, layer := range s.Layers {
		h, out := layer.ForwardSequence(curInput, initH)
		lastH = h
		curInput = out
		initH = nil // only pass initH to first layer
	}
	return lastH, curInput
}

// Parameters returns all parameters.
func (s *StackedGRU) Parameters() []*autograd.Variable {
	var ps []*autograd.Variable
	for _, l := range s.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}

// ZeroGrad zeros all gradients.
func (s *StackedGRU) ZeroGrad() {
	for _, l := range s.Layers {
		l.ZeroGrad()
	}
}
