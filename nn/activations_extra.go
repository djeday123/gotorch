package nn

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── GELU ─────────────────────────────────────────────────────────────────────
// Gaussian Error Linear Unit: x * Φ(x)
// Approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))

type GELULayer struct{}

func NewGELU() *GELULayer                                            { return &GELULayer{} }
func (g *GELULayer) Parameters() []*autograd.Variable                { return nil }
func (g *GELULayer) ZeroGrad()                                        {}
func (g *GELULayer) Forward(x *autograd.Variable) *autograd.Variable { return geluForward(x) }

type geluBackward struct{ xData *tensor.Tensor }

var sqrt2overPi = math.Sqrt(2.0 / math.Pi)

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

// ── LeakyReLU ────────────────────────────────────────────────────────────────

type LeakyReLULayer struct {
	NegSlope float64
}

func NewLeakyReLU(negSlope float64) *LeakyReLULayer      { return &LeakyReLULayer{NegSlope: negSlope} }
func (l *LeakyReLULayer) Parameters() []*autograd.Variable { return nil }
func (l *LeakyReLULayer) ZeroGrad()                         {}
func (l *LeakyReLULayer) Forward(x *autograd.Variable) *autograd.Variable {
	return leakyReLUForward(x, l.NegSlope)
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

// ── ELU ──────────────────────────────────────────────────────────────────────
// ELU(x) = x if x>0, alpha*(exp(x)-1) otherwise

type ELULayer struct {
	Alpha float64
}

func NewELU(alpha float64) *ELULayer              { return &ELULayer{Alpha: alpha} }
func (e *ELULayer) Parameters() []*autograd.Variable { return nil }
func (e *ELULayer) ZeroGrad()                         {}
func (e *ELULayer) Forward(x *autograd.Variable) *autograd.Variable {
	return eluForward(x, e.Alpha)
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

// ── SiLU / Swish ─────────────────────────────────────────────────────────────
// SiLU(x) = x * sigmoid(x)

type SiLULayer struct{}

func NewSiLU() *SiLULayer                                            { return &SiLULayer{} }
func (s *SiLULayer) Parameters() []*autograd.Variable                { return nil }
func (s *SiLULayer) ZeroGrad()                                        {}
func (s *SiLULayer) Forward(x *autograd.Variable) *autograd.Variable { return siluForward(x) }

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

// ── Softplus ─────────────────────────────────────────────────────────────────
// Softplus(x) = (1/beta) * log(1 + exp(beta * x))

type SoftplusLayer struct {
	Beta float64
}

func NewSoftplus(beta float64) *SoftplusLayer      { return &SoftplusLayer{Beta: beta} }
func (s *SoftplusLayer) Parameters() []*autograd.Variable { return nil }
func (s *SoftplusLayer) ZeroGrad()                         {}
func (s *SoftplusLayer) Forward(x *autograd.Variable) *autograd.Variable {
	return softplusForward(x, s.Beta)
}

type softplusBackward struct {
	xData *tensor.Tensor
	beta  float64
}

func (f *softplusBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gd := grad.Data()
	xd := f.xData.Data()
	dxd := make([]float64, len(xd))
	for i, xi := range xd {
		// d/dx log(1+exp(beta*x))/beta = sigmoid(beta*x)
		dxd[i] = gd[i] / (1 + math.Exp(-f.beta*xi))
	}
	return []*tensor.Tensor{tensor.New(dxd, f.xData.Shape())}
}

func softplusForward(x *autograd.Variable, beta float64) *autograd.Variable {
	d := x.Data.Data()
	out := make([]float64, len(d))
	for i, v := range d {
		out[i] = (1.0 / beta) * math.Log(1+math.Exp(beta*v))
	}
	return autograd.NewResult(tensor.New(out, x.Data.Shape()), &softplusBackward{x.Data, beta}, x)
}
