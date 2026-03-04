package optim

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// AdamW implements Adam with decoupled weight decay (Loshchilov & Hutter 2019).
// Unlike Adam, weight decay is applied directly to the parameters before the
// gradient update — not folded into the gradient. This produces better
// regularisation for most architectures.
//
//	opt := optim.NewAdamW(model.Parameters(), 1e-3, 0.9, 0.999, 1e-8, 0.01)
type AdamW struct {
	params      []*autograd.Variable
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	step        int
	m           [][]float64 // first moment
	v           [][]float64 // second moment
}

func NewAdamW(params []*autograd.Variable, lr, beta1, beta2, eps, weightDecay float64) *AdamW {
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	for i, p := range params {
		m[i] = make([]float64, p.Data.Size())
		v[i] = make([]float64, p.Data.Size())
	}
	return &AdamW{params: params, lr: lr, beta1: beta1, beta2: beta2,
		eps: eps, weightDecay: weightDecay, m: m, v: v}
}

func (a *AdamW) ZeroGrad() {
	for _, p := range a.params {
		p.ZeroGrad()
	}
}

func (a *AdamW) Step() {
	a.step++
	bc1 := 1 - math.Pow(a.beta1, float64(a.step))
	bc2 := 1 - math.Pow(a.beta2, float64(a.step))

	for i, p := range a.params {
		if p.Grad == nil {
			continue
		}
		data := p.Data.Data()
		grad := p.Grad.Data()

		for j := range data {
			// Decoupled weight decay (applied before gradient step)
			data[j] *= (1 - a.lr*a.weightDecay)

			g := grad[j]
			a.m[i][j] = a.beta1*a.m[i][j] + (1-a.beta1)*g
			a.v[i][j] = a.beta2*a.v[i][j] + (1-a.beta2)*g*g

			mHat := a.m[i][j] / bc1
			vHat := a.v[i][j] / bc2

			data[j] -= a.lr * mHat / (math.Sqrt(vHat) + a.eps)
		}
		p.Data = tensor.New(data, p.Data.Shape())
	}
}

func (a *AdamW) SetLR(lr float64) { a.lr = lr }
func (a *AdamW) GetLR() float64   { return a.lr }

// AdamWState holds serialisable snapshot of AdamW internal state.
type AdamWState struct {
	Type        string      `json:"type"`
	Step        int         `json:"step"`
	LR          float64     `json:"lr"`
	Beta1       float64     `json:"beta1"`
	Beta2       float64     `json:"beta2"`
	Eps         float64     `json:"eps"`
	WeightDecay float64     `json:"weight_decay"`
	M           [][]float64 `json:"m"`
	V           [][]float64 `json:"v"`
}

func (a *AdamW) GetState() AdamWState {
	return AdamWState{
		Type: "adamw", Step: a.step, LR: a.lr,
		Beta1: a.beta1, Beta2: a.beta2, Eps: a.eps,
		WeightDecay: a.weightDecay, M: a.m, V: a.v,
	}
}

func (a *AdamW) SetState(s AdamWState) {
	a.step = s.Step
	a.lr = s.LR
	a.beta1 = s.Beta1
	a.beta2 = s.Beta2
	a.eps = s.Eps
	a.weightDecay = s.WeightDecay
	a.m = s.M
	a.v = s.V
}
