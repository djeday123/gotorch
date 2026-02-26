package nn

import (
	"gotorch_v1/autograd"
	"gotorch_v1/tensor"
	"math"
	"math/rand"
)

// Linear implements y = x @ W.T + b  (PyTorch convention).
// Weight shape: (outFeatures, inFeatures)
// Bias shape:   (outFeatures,)
type Linear struct {
	Weight  *autograd.Variable
	Bias    *autograd.Variable
	useBias bool
}

// NewLinear creates a Linear layer with Xavier uniform weight initialization.
func NewLinear(inFeatures, outFeatures int, bias bool) *Linear {
	limit := math.Sqrt(6.0 / float64(inFeatures+outFeatures))
	wData := make([]float64, outFeatures*inFeatures)
	for i := range wData {
		wData[i] = (rand.Float64()*2 - 1) * limit
	}
	w := autograd.NewVar(tensor.New(wData, []int{outFeatures, inFeatures}), true)

	var b *autograd.Variable
	if bias {
		b = autograd.NewVar(tensor.Zeros(outFeatures), true)
	}
	return &Linear{Weight: w, Bias: b, useBias: bias}
}

// Forward computes x @ W.T + b.
// Input x: (N, inFeatures) → output: (N, outFeatures)
func (l *Linear) Forward(x *autograd.Variable) *autograd.Variable {
	out := linearMatMul(x, l.Weight)
	if l.useBias && l.Bias != nil {
		out = addBias(out, l.Bias)
	}
	return out
}

// Parameters returns trainable parameters (Weight and optionally Bias).
func (l *Linear) Parameters() []*autograd.Variable {
	if l.useBias && l.Bias != nil {
		return []*autograd.Variable{l.Weight, l.Bias}
	}
	return []*autograd.Variable{l.Weight}
}

// ZeroGrad zeros gradients of all parameters.
func (l *Linear) ZeroGrad() {
	l.Weight.ZeroGrad()
	if l.useBias && l.Bias != nil {
		l.Bias.ZeroGrad()
	}
}

// ---- linearMatMul: x @ w.T with proper gradient flow ----

type linearMatMulBackward struct {
	xData *tensor.Tensor
	wData *tensor.Tensor
}

func (f *linearMatMulBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// grad shape: (N, out)
	// grad_x = grad @ w         → (N, out) @ (out, in) = (N, in)
	// grad_w = grad.T @ x       → (out, N) @ (N, in)  = (out, in)
	gradX := tensor.MatMul(grad, f.wData)
	gradW := tensor.MatMul(grad.T(), f.xData)
	return []*tensor.Tensor{gradX, gradW}
}

func linearMatMul(x, w *autograd.Variable) *autograd.Variable {
	out := tensor.MatMul(x.Data, w.Data.T())
	return autograd.NewResult(out, &linearMatMulBackward{x.Data, w.Data}, x, w)
}

// ---- addBias: mat + b with proper gradient for b ----

type addBiasBackward struct{}

func (f *addBiasBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// grad_mat  = grad
	// grad_bias = sum over batch (axis 0)
	gradB := tensor.Sum(grad, 0, false)
	return []*tensor.Tensor{grad, gradB}
}

func addBias(mat, b *autograd.Variable) *autograd.Variable {
	// b shape: (out,) → reshape to (1, out) for broadcasting
	bRow := b.Data.Reshape(1, b.Data.Size())
	out := tensor.Add(mat.Data, bRow)
	return autograd.NewResult(out, &addBiasBackward{}, mat, b)
}
