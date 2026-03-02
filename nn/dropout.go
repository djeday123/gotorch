package nn

import (
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// Dropout randomly zeroes elements with probability p during training,
// scaling surviving elements by 1/(1-p) (inverted dropout).
// In eval mode (Training=false) it is a no-op.
type Dropout struct {
	P        float64 // drop probability
	Training bool
}

// NewDropout creates a Dropout layer.
// p is the probability of zeroing each element (0 ≤ p < 1).
func NewDropout(p float64) *Dropout {
	if p < 0 || p >= 1 {
		panic("nn.Dropout: p must be in [0, 1)")
	}
	return &Dropout{P: p, Training: true}
}

func (d *Dropout) Train() { d.Training = true }
func (d *Dropout) Eval()  { d.Training = false }

func (d *Dropout) Parameters() []*autograd.Variable { return nil }
func (d *Dropout) ZeroGrad()                         {}

func (d *Dropout) Forward(x *autograd.Variable) *autograd.Variable {
	if !d.Training || d.P == 0 {
		return x
	}
	return dropoutForward(x, d.P)
}

type dropoutBackward struct {
	mask *tensor.Tensor
}

func (f *dropoutBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gradData := grad.Data()
	maskData := f.mask.Data()
	out := make([]float64, len(gradData))
	for i := range out {
		out[i] = gradData[i] * maskData[i]
	}
	return []*tensor.Tensor{tensor.New(out, grad.Shape())}
}

func dropoutForward(x *autograd.Variable, p float64) *autograd.Variable {
	scale := 1.0 / (1.0 - p)
	xFlat := x.Data.Data()
	out := make([]float64, len(xFlat))
	maskData := make([]float64, len(xFlat))

	for i, v := range xFlat {
		if rand.Float64() >= p {
			out[i] = v * scale
			maskData[i] = scale
		}
	}

	outT := tensor.New(out, x.Data.Shape())
	mask := tensor.New(maskData, x.Data.Shape())

	return autograd.NewResult(outT, &dropoutBackward{mask: mask}, x)
}
