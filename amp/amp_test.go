package amp

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── GradScaler basics ─────────────────────────────────────────────────────────

func TestGradScalerGetScale(t *testing.T) {
	s := NewGradScaler(1024.0)
	if s.GetScale() != 1024.0 {
		t.Errorf("scale=%v want 1024", s.GetScale())
	}
}

func TestGradScalerScaleLoss(t *testing.T) {
	s := NewGradScaler(256.0)
	loss := autograd.NewVar(tensor.Scalar(1.0), false)
	scaled := s.Scale(loss)
	if math.Abs(scaled.Data.Item()-256.0) > 1e-10 {
		t.Errorf("scaled loss = %v want 256", scaled.Data.Item())
	}
}

func TestGradScalerDisabled(t *testing.T) {
	s := NewGradScaler(1024.0)
	s.SetEnabled(false)
	loss := autograd.NewVar(tensor.Scalar(1.0), false)
	scaled := s.Scale(loss)
	// When disabled, Scale returns the original variable (no multiplication)
	if scaled.Data.Item() != 1.0 {
		t.Errorf("disabled scaler: loss=%v want 1.0", scaled.Data.Item())
	}
}

func TestGradScalerIsEnabled(t *testing.T) {
	s := NewGradScaler(512.0)
	if !s.IsEnabled() {
		t.Error("expected scaler to be enabled by default")
	}
	s.SetEnabled(false)
	if s.IsEnabled() {
		t.Error("expected scaler to be disabled after SetEnabled(false)")
	}
}

// ── GradScaler.Step — finite grads ───────────────────────────────────────────

type mockOpt struct {
	stepped bool
}

func (m *mockOpt) Step()    { m.stepped = true }
func (m *mockOpt) ZeroGrad() {}

func TestGradScalerStepFinite(t *testing.T) {
	s := NewGradScaler(8.0)
	opt := &mockOpt{}

	// Param with gradient scaled by 8
	p := autograd.NewVar(tensor.Scalar(1.0), true)
	p.Grad = tensor.Scalar(8.0) // pre-scaled

	ok := s.Step(opt, []*autograd.Variable{p})
	if !ok {
		t.Error("Step should return true for finite gradients")
	}
	if !opt.stepped {
		t.Error("optimizer.Step() should have been called")
	}
	// Grad should be unscaled: 8/8 = 1.0
	if math.Abs(p.Grad.Item()-1.0) > 1e-10 {
		t.Errorf("unscaled grad = %v want 1.0", p.Grad.Item())
	}
}

func TestGradScalerStepInf(t *testing.T) {
	s := NewGradScaler(8.0)
	opt := &mockOpt{}

	p := autograd.NewVar(tensor.Scalar(1.0), true)
	p.Grad = tensor.Scalar(math.Inf(1)) // inf gradient

	ok := s.Step(opt, []*autograd.Variable{p})
	if ok {
		t.Error("Step should return false when inf gradient detected")
	}
	if opt.stepped {
		t.Error("optimizer.Step() should NOT have been called on inf gradient")
	}
}

func TestGradScalerStepNaN(t *testing.T) {
	s := NewGradScaler(8.0)
	opt := &mockOpt{}

	p := autograd.NewVar(tensor.Scalar(1.0), true)
	p.Grad = tensor.Scalar(math.NaN())

	ok := s.Step(opt, []*autograd.Variable{p})
	if ok {
		t.Error("Step should return false on NaN gradient")
	}
}

// ── GradScaler.Update ─────────────────────────────────────────────────────────

func TestGradScalerBackoffOnInf(t *testing.T) {
	// When an inf is detected, scale should be halved on next Update
	// (actually backoff happens in Step when inf detected)
	s := NewGradScalerFull(1024.0, 2.0, 0.5, 10)
	opt := &mockOpt{}
	p := autograd.NewVar(tensor.Scalar(1.0), true)
	p.Grad = tensor.Scalar(math.Inf(1))

	s.Step(opt, []*autograd.Variable{p})
	// Scale should be backoff'd: 1024 * 0.5 = 512
	if math.Abs(s.GetScale()-512.0) > 1e-10 {
		t.Errorf("scale after backoff = %v want 512", s.GetScale())
	}
}

func TestGradScalerGrowthOnGoodIters(t *testing.T) {
	s := NewGradScalerFull(1024.0, 2.0, 0.5, 3) // grow after 3 good iters
	opt := &mockOpt{}

	for i := 0; i < 3; i++ {
		p := autograd.NewVar(tensor.Scalar(1.0), true)
		p.Grad = tensor.Scalar(1.0)
		s.Step(opt, []*autograd.Variable{p})
		s.Update()
	}
	// After 3 good iters, scale should double: 1024 → 2048
	if math.Abs(s.GetScale()-2048.0) > 1e-10 {
		t.Errorf("scale after growth = %v want 2048", s.GetScale())
	}
}

// ── End-to-end: scaler in training loop ──────────────────────────────────────

func TestGradScalerEndToEnd(t *testing.T) {
	// Simple parameter update with scaler
	scaler := NewGradScaler(4.0)

	x := autograd.NewVar(tensor.Scalar(1.0), true)
	// Manual "step" — scale the loss, backward, step
	loss := autograd.MulScalar(x, 2.0) // loss = 2*x
	scaledLoss := scaler.Scale(loss)   // scaled = 8*x
	scaledLoss.Backward()              // grad of x = 8

	opt := &mockOpt{}
	ok := scaler.Step(opt, []*autograd.Variable{x})
	if !ok {
		t.Fatal("Step failed for valid gradients")
	}
	// Unscaled grad should be 8/4 = 2.0
	if math.Abs(x.Grad.Item()-2.0) > 1e-10 {
		t.Errorf("unscaled grad = %v want 2.0", x.Grad.Item())
	}
}
