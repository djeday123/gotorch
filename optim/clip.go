package optim

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ClipGradNorm clips the gradient norm of parameters in-place.
// Returns the total norm before clipping.
//
// Equivalent to torch.nn.utils.clip_grad_norm_(params, max_norm).
func ClipGradNorm(params []*autograd.Variable, maxNorm float64) float64 {
	totalNorm := 0.0
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.Data() {
			totalNorm += g * g
		}
	}
	totalNorm = math.Sqrt(totalNorm)

	if totalNorm > maxNorm {
		scale := maxNorm / (totalNorm + 1e-6)
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			grad := p.Grad.Data()
			for i := range grad {
				grad[i] *= scale
			}
			p.Grad = tensor.New(grad, p.Grad.Shape())
		}
	}
	return totalNorm
}

// ClipGradValue clips each gradient element to [-maxVal, maxVal].
// Equivalent to torch.nn.utils.clip_grad_value_(params, clip_value).
func ClipGradValue(params []*autograd.Variable, maxVal float64) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		grad := p.Grad.Data()
		for i, g := range grad {
			if g > maxVal {
				grad[i] = maxVal
			} else if g < -maxVal {
				grad[i] = -maxVal
			}
		}
		p.Grad = tensor.New(grad, p.Grad.Shape())
	}
}
