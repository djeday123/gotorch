package optim

import (
	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// SGD implements Stochastic Gradient Descent with optional momentum.
type SGD struct {
	params   []*autograd.Variable
	lr       float64
	momentum float64
	velocity [][]float64 // velocity buffers for momentum
}

// NewSGD creates a new SGD optimizer.
// lr: learning rate. momentum: 0 disables momentum.
func NewSGD(params []*autograd.Variable, lr, momentum float64) *SGD {
	velocity := make([][]float64, len(params))
	if momentum > 0 {
		for i, p := range params {
			velocity[i] = make([]float64, p.Data.Size())
		}
	}
	return &SGD{params: params, lr: lr, momentum: momentum, velocity: velocity}
}

// Step performs one optimization step: param -= lr * grad (+ momentum).
func (s *SGD) Step() {
	for i, p := range s.params {
		if p.Grad == nil {
			continue
		}
		gFlat := p.Grad.Data()
		pFlat := p.Data.Data()

		if s.momentum > 0 {
			for j := range pFlat {
				s.velocity[i][j] = s.momentum*s.velocity[i][j] + gFlat[j]
				pFlat[j] -= s.lr * s.velocity[i][j]
			}
		} else {
			for j := range pFlat {
				pFlat[j] -= s.lr * gFlat[j]
			}
		}
		p.Data = tensor.New(pFlat, p.Data.Shape())
	}
}

// ZeroGrad zeros all parameter gradients.
func (s *SGD) ZeroGrad() {
	for _, p := range s.params {
		p.Grad = nil
	}
}

func (s *SGD) SetLR(lr float64) { s.lr = lr }
func (s *SGD) GetLR() float64   { return s.lr }
