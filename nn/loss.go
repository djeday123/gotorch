package nn

import (
	"gotorch_v1/autograd"
	"gotorch_v1/tensor"
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
