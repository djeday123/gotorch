package autograd

import (
	"github.com/djeday123/gotorch/tensor"
	"math"
)

// ---- Add ----

type addBackward struct{ aShape, bShape []int }

func (f *addBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{
		unbroadcastGrad(grad, f.aShape),
		unbroadcastGrad(grad, f.bShape),
	}
}

func Add(a, b *Variable) *Variable {
	out := tensor.Add(a.Data, b.Data)
	return newResult(out, &addBackward{a.Data.Shape(), b.Data.Shape()}, a, b)
}

// ---- Sub ----

type subBackward struct{ aShape, bShape []int }

func (f *subBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{
		unbroadcastGrad(grad, f.aShape),
		unbroadcastGrad(tensor.Neg(grad), f.bShape),
	}
}

func Sub(a, b *Variable) *Variable {
	out := tensor.Sub(a.Data, b.Data)
	return newResult(out, &subBackward{a.Data.Shape(), b.Data.Shape()}, a, b)
}

// ---- Mul ----

type mulBackward struct {
	aData, bData *tensor.Tensor
	aShape, bShape []int
}

func (f *mulBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	ga := unbroadcastGrad(tensor.Mul(grad, f.bData), f.aShape)
	gb := unbroadcastGrad(tensor.Mul(grad, f.aData), f.bShape)
	return []*tensor.Tensor{ga, gb}
}

func Mul(a, b *Variable) *Variable {
	out := tensor.Mul(a.Data, b.Data)
	return newResult(out, &mulBackward{a.Data, b.Data, a.Data.Shape(), b.Data.Shape()}, a, b)
}

// ---- Div ----

type divBackward struct {
	aData, bData *tensor.Tensor
	aShape, bShape []int
}

func (f *divBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// d/da (a/b) = 1/b
	// d/db (a/b) = -a/b^2
	ga := unbroadcastGrad(tensor.Div(grad, f.bData), f.aShape)
	gb := unbroadcastGrad(
		tensor.Neg(tensor.Div(tensor.Mul(grad, f.aData), tensor.Mul(f.bData, f.bData))),
		f.bShape,
	)
	return []*tensor.Tensor{ga, gb}
}

func Div(a, b *Variable) *Variable {
	out := tensor.Div(a.Data, b.Data)
	return newResult(out, &divBackward{a.Data, b.Data, a.Data.Shape(), b.Data.Shape()}, a, b)
}

// ---- MatMul ----

type matMulBackward struct{ aData, bData *tensor.Tensor }

func (f *matMulBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// grad_a = grad @ b.T
	// grad_b = a.T @ grad
	ga := tensor.MatMul(grad, f.bData.T())
	gb := tensor.MatMul(f.aData.T(), grad)
	return []*tensor.Tensor{ga, gb}
}

func MatMul(a, b *Variable) *Variable {
	out := tensor.MatMul(a.Data, b.Data)
	return newResult(out, &matMulBackward{a.Data, b.Data}, a, b)
}

// ---- AddScalar ----

type addScalarBackward struct{ shape []int }

func (f *addScalarBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{grad}
}

func AddScalar(a *Variable, s float64) *Variable {
	out := tensor.AddScalar(a.Data, s)
	return newResult(out, &addScalarBackward{a.Data.Shape()}, a)
}

// ---- MulScalar ----

type mulScalarBackward struct{ s float64 }

func (f *mulScalarBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{tensor.MulScalar(grad, f.s)}
}

func MulScalar(a *Variable, s float64) *Variable {
	out := tensor.MulScalar(a.Data, s)
	return newResult(out, &mulScalarBackward{s}, a)
}

// ---- PowScalar ----

type powScalarBackward struct {
	p    float64
	data *tensor.Tensor
}

func (f *powScalarBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// d/dx x^p = p * x^(p-1)
	base := tensor.PowScalar(f.data, f.p-1)
	gx := tensor.Mul(grad, tensor.MulScalar(base, f.p))
	return []*tensor.Tensor{gx}
}

func PowScalar(a *Variable, p float64) *Variable {
	out := tensor.PowScalar(a.Data, p)
	return newResult(out, &powScalarBackward{p, a.Data}, a)
}

// ---- Neg ----

type negBackward struct{}

func (f *negBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{tensor.Neg(grad)}
}

func Neg(a *Variable) *Variable {
	out := tensor.Neg(a.Data)
	return newResult(out, &negBackward{}, a)
}

// ---- Exp ----

type expBackward struct{ outData *tensor.Tensor }

func (f *expBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{tensor.Mul(grad, f.outData)}
}

func Exp(a *Variable) *Variable {
	out := tensor.Exp(a.Data)
	return newResult(out, &expBackward{out}, a)
}

// ---- Log ----

type logBackward struct{ inData *tensor.Tensor }

func (f *logBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{tensor.Div(grad, f.inData)}
}

func Log(a *Variable) *Variable {
	out := tensor.Log(a.Data)
	return newResult(out, &logBackward{a.Data}, a)
}

// ---- Sum (all elements) ----

type sumAllBackward struct{ shape []int }

func (f *sumAllBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// broadcast scalar grad to input shape
	return []*tensor.Tensor{tensor.Ones(f.shape...)}
}

func Sum(a *Variable) *Variable {
	out := tensor.Sum(a.Data, -999, false)
	return newResult(out, &sumAllBackward{a.Data.Shape()}, a)
}

// ---- SumDim (reduce along axis) ----

type sumDimBackward struct {
	shape []int
	dim   int
}

func (f *sumDimBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// Expand grad back along the reduced dim
	g := grad.Unsqueeze(f.dim)
	expanded := tensor.Add(tensor.Zeros(f.shape...), g) // broadcast
	return []*tensor.Tensor{expanded}
}

func SumDim(a *Variable, dim int) *Variable {
	out := tensor.Sum(a.Data, dim, false)
	return newResult(out, &sumDimBackward{a.Data.Shape(), dim}, a)
}

// ---- Mean (all elements) ----

type meanAllBackward struct {
	shape []int
	n     float64
}

func (f *meanAllBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{tensor.MulScalar(tensor.Ones(f.shape...), 1.0/f.n)}
}

func Mean(a *Variable) *Variable {
	out := tensor.Mean(a.Data, -999, false)
	n := float64(a.Data.Size())
	return newResult(out, &meanAllBackward{a.Data.Shape(), n}, a)
}

// ---- ReLU ----

type reluBackward struct{ inData *tensor.Tensor }

func (f *reluBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// mask: 1 where input > 0, else 0
	mask := tensor.Zeros(f.inData.Shape()...)
	flat := f.inData.Data()
	for i, v := range flat {
		if v > 0 {
			mask.Data()[i] = 1
		}
	}
	return []*tensor.Tensor{tensor.Mul(grad, mask)}
}

// maskData returns the underlying flat slice of a tensor (via Data() copy).
// We need a writable view for the mask trick.
func applyMask(grad, inData *tensor.Tensor) *tensor.Tensor {
	out := tensor.Zeros(inData.Shape()...)
	inFlat := inData.Data()
	gradFlat := grad.Data()
	outFlat := out.Data()
	for i, v := range inFlat {
		if v > 0 {
			outFlat[i] = gradFlat[i]
		}
	}
	return tensor.New(outFlat, inData.Shape())
}

func ReLU(a *Variable) *Variable {
	out := tensor.ReLU(a.Data)
	fn := &reluBackward{a.Data}
	_ = fn
	// Use inline closure-based gradFn
	return newResult(out, &reluBackwardFn{a.Data}, a)
}

type reluBackwardFn struct{ inData *tensor.Tensor }

func (f *reluBackwardFn) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{applyMask(grad, f.inData)}
}

// ---- Sigmoid ----

type sigmoidBackward struct{ outData *tensor.Tensor }

func (f *sigmoidBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// s * (1 - s)
	ones := tensor.Ones(f.outData.Shape()...)
	deriv := tensor.Mul(f.outData, tensor.Sub(ones, f.outData))
	return []*tensor.Tensor{tensor.Mul(grad, deriv)}
}

func Sigmoid(a *Variable) *Variable {
	out := tensor.Sigmoid(a.Data)
	return newResult(out, &sigmoidBackward{out}, a)
}

// ---- Tanh ----

type tanhBackward struct{ outData *tensor.Tensor }

func (f *tanhBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// 1 - tanh^2
	ones := tensor.Ones(f.outData.Shape()...)
	deriv := tensor.Sub(ones, tensor.Mul(f.outData, f.outData))
	return []*tensor.Tensor{tensor.Mul(grad, deriv)}
}

func Tanh(a *Variable) *Variable {
	out := tensor.Tanh(a.Data)
	return newResult(out, &tanhBackward{out}, a)
}

// ---- Softmax (for use in loss, not recommended to differentiate directly) ----

func Softmax(a *Variable, dim int) *Variable {
	out := tensor.Softmax(a.Data, dim)
	// Gradient through softmax is complex; for CrossEntropyLoss use LogSoftmax+NLLLoss
	// Here we provide an approximate gradient (identity) — use inside loss only
	return newResult(out, &identityBackward{}, a)
}

type identityBackward struct{}

func (f *identityBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{grad}
}

// ---- numerical gradient helper (exported for tests) ----

// NumericalGrad approximates the gradient of fn w.r.t. x using finite differences.
func NumericalGrad(fn func(*tensor.Tensor) float64, x *tensor.Tensor, eps float64) *tensor.Tensor {
	grad := tensor.Zeros(x.Shape()...)
	xFlat := x.Data()
	gFlat := grad.Data()
	for i := range xFlat {
		orig := xFlat[i]
		xFlat[i] = orig + eps
		fp := fn(tensor.New(xFlat, x.Shape()))
		xFlat[i] = orig - eps
		fm := fn(tensor.New(xFlat, x.Shape()))
		xFlat[i] = orig
		gFlat[i] = (fp - fm) / (2 * eps)
	}
	return tensor.New(gFlat, x.Shape())
}

// absMax returns max absolute value in tensor (for allclose checks).
func AbsMax(t *tensor.Tensor) float64 {
	flat := t.Data()
	max := 0.0
	for _, v := range flat {
		a := math.Abs(v)
		if a > max {
			max = a
		}
	}
	return max
}

// AllClose returns true if all elements differ by less than tol.
func AllClose(a, b *tensor.Tensor, tol float64) bool {
	af, bf := a.Data(), b.Data()
	if len(af) != len(bf) {
		return false
	}
	for i := range af {
		if math.Abs(af[i]-bf[i]) > tol {
			return false
		}
	}
	return true
}
