package optim

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// RMSprop implements the RMSprop optimizer (Hinton, 2012).
//
// Update rule:
//
//	v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
//	θ_t = θ_{t-1} - lr / sqrt(v_t + eps) * g_t
//
// With optional momentum:
//
//	buf_t = momentum * buf_{t-1} + lr / sqrt(v_t + eps) * g_t
//	θ_t   = θ_{t-1} - buf_t
type RMSprop struct {
	params   []*autograd.Variable
	lr       float64
	alpha    float64 // smoothing constant (typically 0.99)
	eps      float64
	momentum float64
	decay    float64 // weight decay (L2)

	v   [][]float64 // running mean square
	buf [][]float64 // momentum buffer
}

// NewRMSprop creates an RMSprop optimizer.
// Typical: lr=0.01, alpha=0.99, eps=1e-8, momentum=0, decay=0
func NewRMSprop(params []*autograd.Variable, lr, alpha, eps, momentum, decay float64) *RMSprop {
	v := make([][]float64, len(params))
	buf := make([][]float64, len(params))
	for i, p := range params {
		v[i] = make([]float64, p.Data.Size())
		buf[i] = make([]float64, p.Data.Size())
	}
	return &RMSprop{params: params, lr: lr, alpha: alpha, eps: eps, momentum: momentum, decay: decay, v: v, buf: buf}
}

func (r *RMSprop) Step() {
	for i, p := range r.params {
		if p.Grad == nil {
			continue
		}
		gFlat := p.Grad.Data()
		pFlat := p.Data.Data()

		for j, g := range gFlat {
			if r.decay != 0 {
				g += r.decay * pFlat[j]
			}
			r.v[i][j] = r.alpha*r.v[i][j] + (1-r.alpha)*g*g
			update := r.lr / (math.Sqrt(r.v[i][j]) + r.eps) * g
			if r.momentum != 0 {
				r.buf[i][j] = r.momentum*r.buf[i][j] + update
				pFlat[j] -= r.buf[i][j]
			} else {
				pFlat[j] -= update
			}
		}
		p.Data = tensor.New(pFlat, p.Data.Shape())
	}
}

func (r *RMSprop) ZeroGrad() {
	for _, p := range r.params {
		p.Grad = nil
	}
}

func (r *RMSprop) SetLR(lr float64) { r.lr = lr }
func (r *RMSprop) GetLR() float64   { return r.lr }

// ── Adadelta ─────────────────────────────────────────────────────────────────
// Adadelta (Zeiler, 2012): no global learning rate needed.
//
//	E[g²]_t = rho * E[g²]_{t-1} + (1-rho) * g_t²
//	Δθ_t    = -sqrt(E[Δθ²]_{t-1} + eps) / sqrt(E[g²]_t + eps) * g_t
//	E[Δθ²]_t = rho * E[Δθ²]_{t-1} + (1-rho) * Δθ_t²
type Adadelta struct {
	params []*autograd.Variable
	lr     float64 // scaling factor (default=1.0)
	rho    float64
	eps    float64
	decay  float64

	eg2   [][]float64 // E[g²]
	edx2  [][]float64 // E[Δθ²]
}

// NewAdadelta creates an Adadelta optimizer.
// Typical: lr=1.0, rho=0.95, eps=1e-6, decay=0
func NewAdadelta(params []*autograd.Variable, lr, rho, eps, decay float64) *Adadelta {
	eg2 := make([][]float64, len(params))
	edx2 := make([][]float64, len(params))
	for i, p := range params {
		eg2[i] = make([]float64, p.Data.Size())
		edx2[i] = make([]float64, p.Data.Size())
	}
	return &Adadelta{params: params, lr: lr, rho: rho, eps: eps, decay: decay, eg2: eg2, edx2: edx2}
}

func (a *Adadelta) Step() {
	for i, p := range a.params {
		if p.Grad == nil {
			continue
		}
		gFlat := p.Grad.Data()
		pFlat := p.Data.Data()

		for j, g := range gFlat {
			if a.decay != 0 {
				g += a.decay * pFlat[j]
			}
			a.eg2[i][j] = a.rho*a.eg2[i][j] + (1-a.rho)*g*g
			dx := -math.Sqrt(a.edx2[i][j]+a.eps) / math.Sqrt(a.eg2[i][j]+a.eps) * g
			a.edx2[i][j] = a.rho*a.edx2[i][j] + (1-a.rho)*dx*dx
			pFlat[j] += a.lr * dx
		}
		p.Data = tensor.New(pFlat, p.Data.Shape())
	}
}

func (a *Adadelta) ZeroGrad() {
	for _, p := range a.params {
		p.Grad = nil
	}
}

func (a *Adadelta) SetLR(lr float64) { a.lr = lr }
func (a *Adadelta) GetLR() float64   { return a.lr }
