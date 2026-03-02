package nn

import "github.com/djeday123/gotorch/autograd"

// ReLULayer is a Module wrapper for the ReLU activation.
type ReLULayer struct{}

func NewReLU() *ReLULayer                                          { return &ReLULayer{} }
func (r *ReLULayer) Forward(x *autograd.Variable) *autograd.Variable { return autograd.ReLU(x) }
func (r *ReLULayer) Parameters() []*autograd.Variable               { return nil }
func (r *ReLULayer) ZeroGrad()                                       {}

// SigmoidLayer is a Module wrapper for the Sigmoid activation.
type SigmoidLayer struct{}

func NewSigmoid() *SigmoidLayer                                        { return &SigmoidLayer{} }
func (s *SigmoidLayer) Forward(x *autograd.Variable) *autograd.Variable { return autograd.Sigmoid(x) }
func (s *SigmoidLayer) Parameters() []*autograd.Variable               { return nil }
func (s *SigmoidLayer) ZeroGrad()                                       {}

// TanhLayer is a Module wrapper for the Tanh activation.
type TanhLayer struct{}

func NewTanh() *TanhLayer                                           { return &TanhLayer{} }
func (t *TanhLayer) Forward(x *autograd.Variable) *autograd.Variable { return autograd.Tanh(x) }
func (t *TanhLayer) Parameters() []*autograd.Variable               { return nil }
func (t *TanhLayer) ZeroGrad()                                       {}
