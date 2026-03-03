// Package amp provides Automatic Mixed Precision (AMP) utilities.
//
// In PyTorch, AMP allows training with float16/bfloat16 to use less memory
// and run faster on Tensor Cores, while keeping float32 master weights.
//
// GoTorch currently computes in float64 (no float16 hardware path), so this
// package provides the GradScaler API for ergonomics / PyTorch parity.
// When GoTorch gains float16 CUDA support, GradScaler will prevent underflow.
//
// Usage:
//
//	scaler := amp.NewGradScaler(1024.0)
//	for batch := range loader {
//	    opt.ZeroGrad()
//	    loss := model.Forward(x)
//	    scaler.Scale(loss).Backward()  // scaled backward
//	    scaler.Step(opt)               // unscales grads, calls opt.Step if no inf/nan
//	    scaler.Update()                // adjusts scale for next iter
//	}
package amp

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// Optimizer is the subset of optim.* needed by GradScaler.
type Optimizer interface {
	Step()
	ZeroGrad()
}

// GradScaler implements loss scaling for mixed-precision training.
// It multiplies the loss by a dynamic scale factor before backward,
// then divides the gradients before the optimizer step.
// If inf/nan is detected in the gradients, the update is skipped
// and the scale is reduced.
type GradScaler struct {
	scale     float64
	growthFactor   float64
	backoffFactor  float64
	growthInterval int   // increase scale every N consecutive good iters
	goodIters      int   // consecutive iters without inf/nan
	enabled        bool
}

// NewGradScaler creates a GradScaler with the given initial scale.
// Typical: initScale=2^10=1024 or 2^16=65536.
func NewGradScaler(initScale float64) *GradScaler {
	return &GradScaler{
		scale:          initScale,
		growthFactor:   2.0,
		backoffFactor:  0.5,
		growthInterval: 2000,
		enabled:        true,
	}
}

// NewGradScalerFull creates a GradScaler with all options.
func NewGradScalerFull(initScale, growthFactor, backoffFactor float64, growthInterval int) *GradScaler {
	return &GradScaler{
		scale:          initScale,
		growthFactor:   growthFactor,
		backoffFactor:  backoffFactor,
		growthInterval: growthInterval,
		enabled:        true,
	}
}

// Scale returns loss * scale_factor as a new Variable.
// Call .Backward() on the result to get scaled gradients.
func (s *GradScaler) Scale(loss *autograd.Variable) *autograd.Variable {
	if !s.enabled {
		return loss
	}
	return autograd.MulScalar(loss, s.scale)
}

// Step unscales the gradients and calls optimizer.Step() if no inf/nan found.
// Returns true if the optimizer step was taken (gradients were finite).
func (s *GradScaler) Step(opt Optimizer, params []*autograd.Variable) bool {
	if !s.enabled {
		opt.Step()
		return true
	}
	// Unscale gradients in-place
	invScale := 1.0 / s.scale
	foundInf := false
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		gData := p.Grad.Data()
		newG := make([]float64, len(gData))
		for i, v := range gData {
			newG[i] = v * invScale
			if math.IsNaN(newG[i]) || math.IsInf(newG[i], 0) {
				foundInf = true
			}
		}
		p.Grad = tensor.New(newG, p.Grad.Shape())
	}

	if foundInf {
		s.goodIters = 0
		s.scale *= s.backoffFactor
		return false
	}
	opt.Step()
	return true
}

// Update adjusts the scale at the end of each iteration.
// Call after Step.
func (s *GradScaler) Update() {
	if !s.enabled {
		return
	}
	s.goodIters++
	if s.goodIters >= s.growthInterval {
		s.scale *= s.growthFactor
		s.goodIters = 0
	}
}

// GetScale returns the current loss scale factor.
func (s *GradScaler) GetScale() float64 { return s.scale }

// SetEnabled enables or disables scaling (disabled = no-op pass-through).
func (s *GradScaler) SetEnabled(v bool) { s.enabled = v }

// IsEnabled reports whether scaling is active.
func (s *GradScaler) IsEnabled() bool { return s.enabled }
