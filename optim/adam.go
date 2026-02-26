package optim

import (
	"gotorch_v1/autograd"
	"gotorch_v1/tensor"
	"math"
)

// Adam implements the Adam optimizer (Kingma & Ba, 2014).
type Adam struct {
	params []*autograd.Variable
	lr     float64
	beta1  float64
	beta2  float64
	eps    float64
	t      int        // step counter
	m      [][]float64 // first moment (mean)
	v      [][]float64 // second moment (variance)
}

// NewAdam creates an Adam optimizer with sensible defaults.
// lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
func NewAdam(params []*autograd.Variable, lr, beta1, beta2, eps float64) *Adam {
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	for i, p := range params {
		m[i] = make([]float64, p.Data.Size())
		v[i] = make([]float64, p.Data.Size())
	}
	return &Adam{params: params, lr: lr, beta1: beta1, beta2: beta2, eps: eps, m: m, v: v}
}

// Step performs one Adam update step.
func (a *Adam) Step() {
	a.t++
	// Bias correction factors
	bc1 := 1 - math.Pow(a.beta1, float64(a.t))
	bc2 := 1 - math.Pow(a.beta2, float64(a.t))
	lrT := a.lr * math.Sqrt(bc2) / bc1

	for i, p := range a.params {
		if p.Grad == nil {
			continue
		}
		gFlat := p.Grad.Data()
		pFlat := p.Data.Data()

		for j, g := range gFlat {
			a.m[i][j] = a.beta1*a.m[i][j] + (1-a.beta1)*g
			a.v[i][j] = a.beta2*a.v[i][j] + (1-a.beta2)*g*g
			pFlat[j] -= lrT * a.m[i][j] / (math.Sqrt(a.v[i][j]) + a.eps)
		}
		p.Data = tensor.New(pFlat, p.Data.Shape())
	}
}

// ZeroGrad zeros all parameter gradients.
func (a *Adam) ZeroGrad() {
	for _, p := range a.params {
		p.Grad = nil
	}
}
