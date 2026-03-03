package optim

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── RMSprop ──────────────────────────────────────────────────────────────────

func TestRMSpropStep(t *testing.T) {
	// Simple: minimize f(x) = x^2 starting from x=2 for a few steps
	x := autograd.NewVar(tensor.Scalar(2.0), true)
	opt := NewRMSprop([]*autograd.Variable{x}, 0.01, 0.99, 1e-8, 0, 0)

	for i := 0; i < 100; i++ {
		opt.ZeroGrad()
		loss := autograd.MulScalar(autograd.PowScalar(x, 2), 1.0)
		loss.Backward()
		opt.Step()
	}
	// Should converge toward 0
	if math.Abs(x.Data.Item()) > 0.5 {
		t.Errorf("RMSprop: x=%v after 100 steps, expected closer to 0", x.Data.Item())
	}
}

func TestRMSpropDecreasing(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(5.0), true)
	opt := NewRMSprop([]*autograd.Variable{x}, 0.1, 0.99, 1e-8, 0, 0)

	prevLoss := math.Inf(1)
	for i := 0; i < 10; i++ {
		opt.ZeroGrad()
		xv := x.Data.Item()
		loss := xv * xv
		// Set grad manually (2*x)
		x.Grad = tensor.Scalar(2 * xv)
		opt.Step()
		if loss > prevLoss+1e-10 {
			t.Errorf("step %d: loss increased %v→%v", i, prevLoss, loss)
		}
		prevLoss = loss
	}
}

func TestRMSpropMomentum(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(3.0), true)
	opt := NewRMSprop([]*autograd.Variable{x}, 0.01, 0.99, 1e-8, 0.9, 0)

	for i := 0; i < 100; i++ {
		opt.ZeroGrad()
		xv := x.Data.Item()
		x.Grad = tensor.Scalar(2 * xv)
		opt.Step()
	}
	if math.Abs(x.Data.Item()) > 0.5 {
		t.Errorf("RMSprop+momentum: x=%v expected closer to 0", x.Data.Item())
	}
}

func TestRMSpropWeightDecay(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(2.0), true)
	opt := NewRMSprop([]*autograd.Variable{x}, 0.01, 0.99, 1e-8, 0, 0.1)

	for i := 0; i < 50; i++ {
		opt.ZeroGrad()
		xv := x.Data.Item()
		x.Grad = tensor.Scalar(2 * xv) // grad of x^2
		opt.Step()
	}
	// Weight decay adds L2 regularization — should converge faster or to smaller value
	if math.Abs(x.Data.Item()) > 1.0 {
		t.Errorf("RMSprop+decay: x=%v expected smaller", x.Data.Item())
	}
}

func TestRMSpropZeroGrad(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(1.0), true)
	x.Grad = tensor.Scalar(3.0)
	opt := NewRMSprop([]*autograd.Variable{x}, 0.01, 0.99, 1e-8, 0, 0)
	opt.ZeroGrad()
	if x.Grad != nil {
		t.Error("ZeroGrad should set Grad to nil")
	}
}

func TestRMSpropGetSetLR(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(1.0), true)
	opt := NewRMSprop([]*autograd.Variable{x}, 0.01, 0.99, 1e-8, 0, 0)
	if opt.GetLR() != 0.01 {
		t.Errorf("GetLR=%v want 0.01", opt.GetLR())
	}
	opt.SetLR(0.001)
	if opt.GetLR() != 0.001 {
		t.Errorf("after SetLR: GetLR=%v want 0.001", opt.GetLR())
	}
}

// ── Adadelta ─────────────────────────────────────────────────────────────────

func TestAdadeltaStep(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(3.0), true)
	opt := NewAdadelta([]*autograd.Variable{x}, 1.0, 0.95, 1e-6, 0)

	for i := 0; i < 2000; i++ {
		opt.ZeroGrad()
		xv := x.Data.Item()
		x.Grad = tensor.Scalar(2 * xv) // grad of x^2
		opt.Step()
	}
	// Adadelta is slower than Adam — check it makes meaningful progress
	if math.Abs(x.Data.Item()) > 2.5 {
		t.Errorf("Adadelta: x=%v after 2000 steps, expected smaller than 2.5", x.Data.Item())
	}
}

func TestAdadeltaDecreasing(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(4.0), true)
	opt := NewAdadelta([]*autograd.Variable{x}, 1.0, 0.95, 1e-6, 0)

	prevLoss := 16.0
	for i := 0; i < 20; i++ {
		opt.ZeroGrad()
		xv := x.Data.Item()
		loss := xv * xv
		x.Grad = tensor.Scalar(2 * xv)
		opt.Step()
		if i > 2 && loss > prevLoss+1.0 {
			t.Errorf("step %d: loss increased significantly %v→%v", i, prevLoss, loss)
		}
		prevLoss = loss
	}
}

func TestAdadeltaZeroGrad(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(1.0), true)
	x.Grad = tensor.Scalar(5.0)
	opt := NewAdadelta([]*autograd.Variable{x}, 1.0, 0.95, 1e-6, 0)
	opt.ZeroGrad()
	if x.Grad != nil {
		t.Error("ZeroGrad should nil the gradient")
	}
}

func TestAdadeltaGetSetLR(t *testing.T) {
	x := autograd.NewVar(tensor.Scalar(1.0), true)
	opt := NewAdadelta([]*autograd.Variable{x}, 1.0, 0.95, 1e-6, 0)
	opt.SetLR(0.5)
	if opt.GetLR() != 0.5 {
		t.Errorf("GetLR=%v want 0.5", opt.GetLR())
	}
}
