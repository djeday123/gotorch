// Package functional provides stateless (functional) versions of neural network operations,
// mirroring PyTorch's torch.nn.functional API.
package functional

import (
	"math"
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── Activations ──────────────────────────────────────────────────────────────

// ReLU applies the rectified linear unit function element-wise.
func ReLU(x *autograd.Variable) *autograd.Variable { return autograd.ReLU(x) }

// Sigmoid applies the sigmoid function element-wise.
func Sigmoid(x *autograd.Variable) *autograd.Variable { return autograd.Sigmoid(x) }

// Tanh applies the hyperbolic tangent function element-wise.
func Tanh(x *autograd.Variable) *autograd.Variable { return autograd.Tanh(x) }

// GELU applies the Gaussian Error Linear Unit activation.
func GELU(x *autograd.Variable) *autograd.Variable { return geluForward(x) }

// LeakyReLU applies Leaky ReLU with the given negative slope.
func LeakyReLU(x *autograd.Variable, negativeSlope float64) *autograd.Variable {
	return leakyReLUForward(x, negativeSlope)
}

// ELU applies the Exponential Linear Unit activation.
func ELU(x *autograd.Variable, alpha float64) *autograd.Variable {
	return eluForward(x, alpha)
}

// SiLU applies the Sigmoid-weighted Linear Unit (Swish) activation.
func SiLU(x *autograd.Variable) *autograd.Variable { return siluForward(x) }

// Softmax applies softmax along the given dimension.
func Softmax(x *autograd.Variable, dim int) *autograd.Variable {
	return autograd.Softmax(x, dim)
}

// LogSoftmax applies log(softmax(x)) in a numerically stable way.
func LogSoftmax(x *autograd.Variable, dim int) *autograd.Variable {
	return logSoftmaxForward(x, dim)
}

// ── Dropout ──────────────────────────────────────────────────────────────────

// Dropout randomly zeroes elements with probability p during training.
// When training=false, returns x unchanged.
func Dropout(x *autograd.Variable, p float64, training bool) *autograd.Variable {
	if !training || p == 0 {
		return x
	}
	return dropoutForward(x, p)
}

// ── Loss functions ────────────────────────────────────────────────────────────

// MSELoss computes mean squared error: mean((pred - target)^2).
func MSELoss(pred, target *autograd.Variable) *autograd.Variable {
	diff := autograd.Sub(pred, target)
	sq := autograd.Mul(diff, diff)
	return autograd.Mean(sq)
}

// L1Loss computes mean absolute error: mean(|pred - target|).
func L1Loss(pred, target *autograd.Variable) *autograd.Variable {
	return l1LossForward(pred, target)
}

// HuberLoss computes the Huber (smooth L1) loss with the given delta.
// For |x| <= delta: 0.5 * x^2 / delta
// For |x| > delta:  |x| - 0.5 * delta
func HuberLoss(pred, target *autograd.Variable, delta float64) *autograd.Variable {
	return huberLossForward(pred, target, delta)
}

// BCELoss computes binary cross-entropy loss.
// pred must be in (0,1) — apply Sigmoid first.
func BCELoss(pred, target *autograd.Variable) *autograd.Variable {
	eps := 1e-7
	ones := autograd.NewVar(tensor.Ones(pred.Data.Shape()...), false)
	clampedPred := autograd.AddScalar(autograd.MulScalar(pred, 1-2*eps), eps)
	logP := autograd.Log(clampedPred)
	logOneMinusP := autograd.Log(autograd.Sub(ones, clampedPred))
	term1 := autograd.Mul(target, logP)
	term2 := autograd.Mul(autograd.Sub(ones, target), logOneMinusP)
	return autograd.MulScalar(autograd.Mean(autograd.Add(term1, term2)), -1)
}

// CrossEntropyLoss computes cross-entropy from raw logits and integer targets.
// logits: [N, C], targets: class indices [N].
func CrossEntropyLoss(logits *autograd.Variable, targets []int) *autograd.Variable {
	logSoft := logSoftmaxForward(logits, 1)
	return nllLossForward(logSoft, targets)
}

// NLLLoss computes negative log-likelihood loss.
// logProbs: [N, C] (output of LogSoftmax), targets: class indices [N].
func NLLLoss(logProbs *autograd.Variable, targets []int) *autograd.Variable {
	return nllLossForward(logProbs, targets)
}

// ── Linear ───────────────────────────────────────────────────────────────────

// Linear applies a linear transformation: output = x @ weight^T + bias.
// weight: [outFeatures, inFeatures], bias: [outFeatures] (may be nil).
func Linear(x, weight *autograd.Variable, bias *autograd.Variable) *autograd.Variable {
	out := linearMatMul(x, weight)
	if bias != nil {
		out = addBiasF(out, bias)
	}
	return out
}

// ── Internal implementations ─────────────────────────────────────────────────

var sqrt2overPi = math.Sqrt(2.0 / math.Pi)

type geluBackward struct{ xData *tensor.Tensor }

func (f *geluBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gd := grad.Data()
	xd := f.xData.Data()
	dxd := make([]float64, len(xd))
	for i, xi := range xd {
		cdf := 0.5 * (1.0 + math.Tanh(sqrt2overPi*(xi+0.044715*xi*xi*xi)))
		pdf := math.Exp(-0.5*xi*xi) / math.Sqrt(2*math.Pi)
		dxd[i] = gd[i] * (cdf + xi*pdf)
	}
	return []*tensor.Tensor{tensor.New(dxd, f.xData.Shape())}
}

func geluForward(x *autograd.Variable) *autograd.Variable {
	d := x.Data.Data()
	out := make([]float64, len(d))
	for i, v := range d {
		cdf := 0.5 * (1.0 + math.Tanh(sqrt2overPi*(v+0.044715*v*v*v)))
		out[i] = v * cdf
	}
	return autograd.NewResult(tensor.New(out, x.Data.Shape()), &geluBackward{x.Data}, x)
}

type leakyReLUBackward struct {
	xData    *tensor.Tensor
	negSlope float64
}

func (f *leakyReLUBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gd := grad.Data()
	xd := f.xData.Data()
	dxd := make([]float64, len(xd))
	for i, xi := range xd {
		if xi >= 0 {
			dxd[i] = gd[i]
		} else {
			dxd[i] = gd[i] * f.negSlope
		}
	}
	return []*tensor.Tensor{tensor.New(dxd, f.xData.Shape())}
}

func leakyReLUForward(x *autograd.Variable, neg float64) *autograd.Variable {
	d := x.Data.Data()
	out := make([]float64, len(d))
	for i, v := range d {
		if v >= 0 {
			out[i] = v
		} else {
			out[i] = neg * v
		}
	}
	return autograd.NewResult(tensor.New(out, x.Data.Shape()), &leakyReLUBackward{x.Data, neg}, x)
}

type eluBackward struct {
	xData *tensor.Tensor
	alpha float64
}

func (f *eluBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gd := grad.Data()
	xd := f.xData.Data()
	dxd := make([]float64, len(xd))
	for i, xi := range xd {
		if xi >= 0 {
			dxd[i] = gd[i]
		} else {
			dxd[i] = gd[i] * f.alpha * math.Exp(xi)
		}
	}
	return []*tensor.Tensor{tensor.New(dxd, f.xData.Shape())}
}

func eluForward(x *autograd.Variable, alpha float64) *autograd.Variable {
	d := x.Data.Data()
	out := make([]float64, len(d))
	for i, v := range d {
		if v >= 0 {
			out[i] = v
		} else {
			out[i] = alpha * (math.Exp(v) - 1)
		}
	}
	return autograd.NewResult(tensor.New(out, x.Data.Shape()), &eluBackward{x.Data, alpha}, x)
}

type siluBackward struct{ xData *tensor.Tensor }

func (f *siluBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gd := grad.Data()
	xd := f.xData.Data()
	dxd := make([]float64, len(xd))
	for i, xi := range xd {
		sig := 1.0 / (1.0 + math.Exp(-xi))
		dxd[i] = gd[i] * sig * (1 + xi*(1-sig))
	}
	return []*tensor.Tensor{tensor.New(dxd, f.xData.Shape())}
}

func siluForward(x *autograd.Variable) *autograd.Variable {
	d := x.Data.Data()
	out := make([]float64, len(d))
	for i, v := range d {
		sig := 1.0 / (1.0 + math.Exp(-v))
		out[i] = v * sig
	}
	return autograd.NewResult(tensor.New(out, x.Data.Shape()), &siluBackward{x.Data}, x)
}

// logSoftmaxForward: numerically stable log(softmax(x)) along dim.
func logSoftmaxForward(x *autograd.Variable, dim int) *autograd.Variable {
	maxVals := tensor.Max(x.Data, dim, true)
	maxVar := autograd.NewVar(maxVals, false)
	shifted := autograd.Sub(x, maxVar)
	expShifted := autograd.Exp(shifted)
	sumExp := autograd.SumDim(expShifted, dim)
	logSumExp := autograd.Log(sumExp)
	logSumExpExp := autograd.NewVar(logSumExp.Data.Unsqueeze(dim), logSumExp.RequiresGrad)
	return autograd.Sub(shifted, logSumExpExp)
}

// nllLossForward: NLL loss from log-probabilities.
type nllBackwardF struct{ grad *tensor.Tensor }

func (f *nllBackwardF) Apply(upstreamGrad *tensor.Tensor) []*tensor.Tensor {
	scale := upstreamGrad.Item()
	return []*tensor.Tensor{tensor.MulScalar(f.grad, scale)}
}

func nllLossForward(logProbs *autograd.Variable, targets []int) *autograd.Variable {
	n := len(targets)
	c := logProbs.Data.Shape()[1]
	total := 0.0
	for i, t := range targets {
		total += logProbs.Data.At(i, t)
	}
	loss := -total / float64(n)
	gradData := make([]float64, n*c)
	for i, t := range targets {
		gradData[i*c+t] = -1.0 / float64(n)
	}
	gradTensor := tensor.New(gradData, []int{n, c})
	return autograd.NewResult(tensor.Scalar(loss), &nllBackwardF{gradTensor}, logProbs)
}

// L1Loss forward.
type l1LossBackward struct {
	diff *tensor.Tensor
	n    int
}

func (f *l1LossBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	scale := grad.Item() / float64(f.n)
	gd := f.diff.Data()
	out := make([]float64, len(gd))
	for i, v := range gd {
		if v > 0 {
			out[i] = scale
		} else if v < 0 {
			out[i] = -scale
		}
	}
	return []*tensor.Tensor{tensor.New(out, f.diff.Shape()), tensor.New(negSlice(out), f.diff.Shape())}
}

func negSlice(s []float64) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = -v
	}
	return out
}

func l1LossForward(pred, target *autograd.Variable) *autograd.Variable {
	predD := pred.Data.Data()
	targD := target.Data.Data()
	n := len(predD)
	diffD := make([]float64, n)
	sum := 0.0
	for i := range predD {
		d := predD[i] - targD[i]
		diffD[i] = d
		if d < 0 {
			sum -= d
		} else {
			sum += d
		}
	}
	loss := sum / float64(n)
	diffT := tensor.New(diffD, pred.Data.Shape())
	return autograd.NewResult(tensor.Scalar(loss), &l1LossBackward{diffT, n}, pred, target)
}

// HuberLoss forward.
type huberLossBackward struct {
	diff  *tensor.Tensor
	delta float64
	n     int
}

func (f *huberLossBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	scale := grad.Item() / float64(f.n)
	gd := f.diff.Data()
	out := make([]float64, len(gd))
	for i, v := range gd {
		if math.Abs(v) <= f.delta {
			out[i] = scale * v / f.delta
		} else if v > 0 {
			out[i] = scale
		} else {
			out[i] = -scale
		}
	}
	return []*tensor.Tensor{tensor.New(out, f.diff.Shape()), tensor.New(negSlice(out), f.diff.Shape())}
}

func huberLossForward(pred, target *autograd.Variable, delta float64) *autograd.Variable {
	predD := pred.Data.Data()
	targD := target.Data.Data()
	n := len(predD)
	diffD := make([]float64, n)
	sum := 0.0
	for i := range predD {
		d := predD[i] - targD[i]
		diffD[i] = d
		abs := math.Abs(d)
		if abs <= delta {
			sum += 0.5 * d * d / delta
		} else {
			sum += abs - 0.5*delta
		}
	}
	diffT := tensor.New(diffD, pred.Data.Shape())
	return autograd.NewResult(tensor.Scalar(sum/float64(n)), &huberLossBackward{diffT, delta, n}, pred, target)
}

// dropoutForward (inverted dropout, training only).
type dropoutBackwardF struct{ mask *tensor.Tensor }

func (f *dropoutBackwardF) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gd := grad.Data()
	md := f.mask.Data()
	out := make([]float64, len(gd))
	for i := range out {
		out[i] = gd[i] * md[i]
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
	return autograd.NewResult(outT, &dropoutBackwardF{mask}, x)
}

// linearMatMul: x @ weight^T with gradient support.
type linearMatMulBackwardF struct {
	xData *tensor.Tensor
	wData *tensor.Tensor
}

func (f *linearMatMulBackwardF) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gradX := tensor.MatMul(grad, f.wData)
	gradW := tensor.MatMul(grad.T(), f.xData)
	return []*tensor.Tensor{gradX, gradW}
}

func linearMatMul(x, w *autograd.Variable) *autograd.Variable {
	out := tensor.MatMul(x.Data, w.Data.T())
	return autograd.NewResult(out, &linearMatMulBackwardF{x.Data, w.Data}, x, w)
}

// addBiasF adds bias [outF] to mat [N, outF].
type addBiasFBackward struct{}

func (f *addBiasFBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gradB := tensor.Sum(grad, 0, false)
	return []*tensor.Tensor{grad, gradB}
}

func addBiasF(mat, b *autograd.Variable) *autograd.Variable {
	bRow := b.Data.Reshape(1, b.Data.Size())
	out := tensor.Add(mat.Data, bRow)
	return autograd.NewResult(out, &addBiasFBackward{}, mat, b)
}
