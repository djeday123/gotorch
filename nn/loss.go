package nn

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// MSELoss computes the mean squared error: mean((pred - target)^2).
func MSELoss(pred, target *autograd.Variable) *autograd.Variable {
	diff := autograd.Sub(pred, target)
	sq := autograd.Mul(diff, diff)
	return autograd.Mean(sq)
}

// BCELoss computes binary cross-entropy: -mean(t*log(p) + (1-t)*log(1-p)).
// pred must be in (0, 1) — apply Sigmoid before this.
func BCELoss(pred, target *autograd.Variable) *autograd.Variable {
	// Clamp pred to avoid log(0)
	eps := 1e-7
	ones := autograd.NewVar(tensor.Ones(pred.Data.Shape()...), false)
	clampedPred := autograd.AddScalar(autograd.MulScalar(pred, 1-2*eps), eps) // map (0,1) → (eps,1-eps)

	logP := autograd.Log(clampedPred)
	logOneMinusP := autograd.Log(autograd.Sub(ones, clampedPred))

	term1 := autograd.Mul(target, logP)
	term2 := autograd.Mul(autograd.Sub(ones, target), logOneMinusP)

	return autograd.MulScalar(autograd.Mean(autograd.Add(term1, term2)), -1)
}

// CrossEntropyLoss computes cross-entropy loss from raw logits and integer class targets.
// logits shape: (N, C), targets shape: (N,) with values in [0, C).
// Uses LogSoftmax + NLLLoss for numerical stability.
func CrossEntropyLoss(logits *autograd.Variable, targets []int) *autograd.Variable {
	logSoft := logSoftmax(logits, 1) // (N, C)
	return nllLoss(logSoft, targets)
}

// logSoftmax computes log(softmax(x)) in a numerically stable way.
// Returns a Variable with the same shape as x.
func logSoftmax(x *autograd.Variable, dim int) *autograd.Variable {
	maxVals := tensor.Max(x.Data, dim, true)
	maxVar := autograd.NewVar(maxVals, false)
	shifted := autograd.Sub(x, maxVar)
	expShifted := autograd.Exp(shifted)
	sumExp := autograd.SumDim(expShifted, dim)
	logSumExp := autograd.Log(sumExp)
	logSumExpExp := autograd.NewVar(logSumExp.Data.Unsqueeze(dim), logSumExp.RequiresGrad)
	return autograd.Sub(shifted, logSumExpExp)
}

// nllLoss computes Negative Log-Likelihood loss.
// logProbs shape: (N, C), targets: class indices (N,).
func nllLoss(logProbs *autograd.Variable, targets []int) *autograd.Variable {
	n := len(targets)
	c := logProbs.Data.Shape()[1]
	total := 0.0
	for i, t := range targets {
		total += logProbs.Data.At(i, t)
	}
	loss := -total / float64(n)

	// Build gradient: d/d(logProbs[i,j]) = -1/N if j == targets[i], else 0
	gradData := make([]float64, n*c)
	for i, t := range targets {
		gradData[i*c+t] = -1.0 / float64(n)
	}
	gradTensor := tensor.New(gradData, []int{n, c})

	// Use a custom gradFn that injects gradTensor into logProbs
	return autograd.NewResult(tensor.Scalar(loss), &nllBackward{gradTensor}, logProbs)
}

type nllBackward struct{ grad *tensor.Tensor }

func (f *nllBackward) Apply(upstreamGrad *tensor.Tensor) []*tensor.Tensor {
	// upstream is a scalar; scale our precomputed gradient
	scale := upstreamGrad.Item()
	return []*tensor.Tensor{tensor.MulScalar(f.grad, scale)}
}

// NLLLoss computes Negative Log-Likelihood loss (exported version).
// logProbs shape: (N, C), targets: class indices (N,).
func NLLLoss(logProbs *autograd.Variable, targets []int) *autograd.Variable {
	return nllLoss(logProbs, targets)
}

// L1Loss computes mean absolute error: mean(|pred - target|).
func L1Loss(pred, target *autograd.Variable) *autograd.Variable {
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
	diffT := tensor.New(diffD, pred.Data.Shape())
	return autograd.NewResult(tensor.Scalar(sum/float64(n)), &l1LossBackward{diffT, n}, pred, target)
}

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
	neg := make([]float64, len(out))
	for i, v := range out {
		neg[i] = -v
	}
	return []*tensor.Tensor{tensor.New(out, f.diff.Shape()), tensor.New(neg, f.diff.Shape())}
}

// HuberLoss (smooth L1) loss with delta parameter.
// For |x| <= delta: 0.5 * x^2 / delta
// For |x| > delta:  |x| - 0.5 * delta
func HuberLoss(pred, target *autograd.Variable, delta float64) *autograd.Variable {
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
	neg := make([]float64, len(out))
	for i, v := range out {
		neg[i] = -v
	}
	return []*tensor.Tensor{tensor.New(out, f.diff.Shape()), tensor.New(neg, f.diff.Shape())}
}

// KLDivLoss computes element-wise KL divergence: mean(target * (log(target) - input)).
// input should be log-probabilities (e.g. output of LogSoftmax).
// target should be probabilities (positive values summing to 1).
func KLDivLoss(input, target *autograd.Variable) *autograd.Variable {
	inputD := input.Data.Data()
	targetD := target.Data.Data()
	n := len(inputD)
	sum := 0.0
	for i := range inputD {
		t := targetD[i]
		if t > 0 {
			sum += t * (math.Log(t) - inputD[i])
		}
	}
	loss := sum / float64(n)
	// Gradient w.r.t. input: -target / n
	gradIn := make([]float64, n)
	for i, t := range targetD {
		gradIn[i] = -t / float64(n)
	}
	return autograd.NewResult(tensor.Scalar(loss), &klDivBackward{tensor.New(gradIn, input.Data.Shape())}, input)
}

type klDivBackward struct{ grad *tensor.Tensor }

func (f *klDivBackward) Apply(upstream *tensor.Tensor) []*tensor.Tensor {
	scale := upstream.Item()
	return []*tensor.Tensor{tensor.MulScalar(f.grad, scale)}
}
