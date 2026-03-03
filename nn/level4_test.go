package nn

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── GELU ─────────────────────────────────────────────────────────────────────

func TestGELUShape(t *testing.T) {
	g := NewGELU()
	x := autograd.NewVar(tensor.RandN(4, 8), false)
	out := g.Forward(x)
	if out.Data.Shape()[0] != 4 || out.Data.Shape()[1] != 8 {
		t.Fatalf("wrong shape: %v", out.Data.Shape())
	}
}

func TestGELUValues(t *testing.T) {
	// GELU(0) = 0, GELU(large positive) ≈ large positive
	g := NewGELU()
	x := autograd.NewVar(tensor.New([]float64{0, 100.0, -100.0}, []int{3}), false)
	out := g.Forward(x)
	d := out.Data.Data()
	if math.Abs(d[0]) > 1e-6 {
		t.Errorf("GELU(0) = %v want ~0", d[0])
	}
	if math.Abs(d[1]-100.0) > 0.01 {
		t.Errorf("GELU(100) = %v want ~100", d[1])
	}
	if d[2] > 0.01 {
		t.Errorf("GELU(-100) = %v want ~0", d[2])
	}
}

func TestGELUBackward(t *testing.T) {
	g := NewGELU()
	x := autograd.NewVar(tensor.New([]float64{1.0}, []int{1}), true)
	out := g.Forward(x)
	loss := autograd.Sum(out)
	loss.Backward()
	if x.Grad == nil {
		t.Fatal("GELU: x.Grad is nil after backward")
	}
}

// ── LeakyReLU ────────────────────────────────────────────────────────────────

func TestLeakyReLUPositive(t *testing.T) {
	l := NewLeakyReLU(0.1)
	x := autograd.NewVar(tensor.New([]float64{2.0}, []int{1}), false)
	out := l.Forward(x)
	if math.Abs(out.Data.Data()[0]-2.0) > 1e-10 {
		t.Errorf("got %v want 2.0", out.Data.Data()[0])
	}
}

func TestLeakyReLUNegative(t *testing.T) {
	l := NewLeakyReLU(0.1)
	x := autograd.NewVar(tensor.New([]float64{-3.0}, []int{1}), false)
	out := l.Forward(x)
	if math.Abs(out.Data.Data()[0]-(-0.3)) > 1e-10 {
		t.Errorf("got %v want -0.3", out.Data.Data()[0])
	}
}

func TestLeakyReLUBackward(t *testing.T) {
	l := NewLeakyReLU(0.2)
	x := autograd.NewVar(tensor.New([]float64{-1.0}, []int{1}), true)
	out := l.Forward(x)
	out.Backward()
	if math.Abs(x.Grad.Data()[0]-0.2) > 1e-10 {
		t.Errorf("grad = %v want 0.2", x.Grad.Data()[0])
	}
}

// ── ELU ──────────────────────────────────────────────────────────────────────

func TestELUPositive(t *testing.T) {
	e := NewELU(1.0)
	x := autograd.NewVar(tensor.New([]float64{2.0}, []int{1}), false)
	out := e.Forward(x)
	if math.Abs(out.Data.Data()[0]-2.0) > 1e-10 {
		t.Errorf("got %v want 2.0", out.Data.Data()[0])
	}
}

func TestELUNegative(t *testing.T) {
	e := NewELU(1.0)
	x := autograd.NewVar(tensor.New([]float64{-1.0}, []int{1}), false)
	out := e.Forward(x)
	want := math.Exp(-1) - 1 // ≈ -0.632
	if math.Abs(out.Data.Data()[0]-want) > 1e-10 {
		t.Errorf("got %v want %v", out.Data.Data()[0], want)
	}
}

func TestELUBackward(t *testing.T) {
	e := NewELU(1.0)
	x := autograd.NewVar(tensor.New([]float64{-1.0}, []int{1}), true)
	out := e.Forward(x)
	out.Backward()
	want := math.Exp(-1.0) // alpha * exp(x) at x=-1
	if math.Abs(x.Grad.Data()[0]-want) > 1e-10 {
		t.Errorf("ELU grad = %v want %v", x.Grad.Data()[0], want)
	}
}

// ── SiLU ─────────────────────────────────────────────────────────────────────

func TestSiLUZero(t *testing.T) {
	s := NewSiLU()
	x := autograd.NewVar(tensor.New([]float64{0.0}, []int{1}), false)
	out := s.Forward(x)
	if math.Abs(out.Data.Data()[0]) > 1e-10 {
		t.Errorf("SiLU(0) = %v want 0", out.Data.Data()[0])
	}
}

func TestSiLUPositive(t *testing.T) {
	s := NewSiLU()
	x := autograd.NewVar(tensor.New([]float64{1.0}, []int{1}), false)
	out := s.Forward(x)
	// SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
	want := 1.0 / (1 + math.Exp(-1.0))
	if math.Abs(out.Data.Data()[0]-want) > 1e-10 {
		t.Errorf("SiLU(1) = %v want %v", out.Data.Data()[0], want)
	}
}

func TestSiLUBackward(t *testing.T) {
	s := NewSiLU()
	x := autograd.NewVar(tensor.New([]float64{1.0}, []int{1}), true)
	out := s.Forward(x)
	out.Backward()
	if x.Grad == nil {
		t.Fatal("SiLU: x.Grad is nil")
	}
}

// ── Softplus ─────────────────────────────────────────────────────────────────

func TestSoftplusPositive(t *testing.T) {
	sp := NewSoftplus(1.0)
	x := autograd.NewVar(tensor.New([]float64{0.0}, []int{1}), false)
	out := sp.Forward(x)
	want := math.Log(2.0)
	if math.Abs(out.Data.Data()[0]-want) > 1e-10 {
		t.Errorf("Softplus(0) = %v want %v", out.Data.Data()[0], want)
	}
}

func TestSoftplusApproxReLU(t *testing.T) {
	// For large x, Softplus(x) ≈ x
	sp := NewSoftplus(1.0)
	x := autograd.NewVar(tensor.New([]float64{100.0}, []int{1}), false)
	out := sp.Forward(x)
	if math.Abs(out.Data.Data()[0]-100.0) > 1e-3 {
		t.Errorf("Softplus(100) = %v want ~100", out.Data.Data()[0])
	}
}

// ── Conv1d ───────────────────────────────────────────────────────────────────

func TestConv1dShape(t *testing.T) {
	c := NewConv1d(4, 8, 3, 1, 0, true)
	// Input: [N=2, inC=4, L=10] → output: [2, 8, 8]
	x := autograd.NewVar(tensor.RandN(2, 4, 10), false)
	out := c.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 2 || shape[1] != 8 || shape[2] != 8 {
		t.Fatalf("expected [2 8 8], got %v", shape)
	}
}

func TestConv1dWithPadding(t *testing.T) {
	// With padding=1, kernel=3, stride=1: oL = L + 2*1 - 3 + 1 = L
	c := NewConv1d(2, 4, 3, 1, 1, false)
	x := autograd.NewVar(tensor.RandN(1, 2, 16), false)
	out := c.Forward(x)
	if out.Data.Shape()[2] != 16 {
		t.Fatalf("expected oL=16, got %v", out.Data.Shape()[2])
	}
}

func TestConv1dParameters(t *testing.T) {
	c := NewConv1d(3, 8, 5, 1, 0, true)
	params := c.Parameters()
	if len(params) != 2 {
		t.Fatalf("expected 2 params (weight+bias), got %d", len(params))
	}
	// weight: [8, 3, 5]
	wShape := params[0].Data.Shape()
	if wShape[0] != 8 || wShape[1] != 3 || wShape[2] != 5 {
		t.Fatalf("weight shape %v, expected [8 3 5]", wShape)
	}
}

func TestConv1dBackward(t *testing.T) {
	c := NewConv1d(2, 4, 3, 1, 0, false)
	x := autograd.NewVar(tensor.RandN(2, 2, 8), true)
	out := c.Forward(x)
	loss := autograd.Mean(autograd.Sum(out))
	loss.Backward()
	if x.Grad == nil {
		t.Fatal("Conv1d: x.Grad is nil after backward")
	}
	if c.Weight.Grad == nil {
		t.Fatal("Conv1d: weight.Grad is nil after backward")
	}
}

// ── AdaptiveAvgPool2d ─────────────────────────────────────────────────────────

func TestAdaptiveAvgPool2dShape(t *testing.T) {
	pool := NewAdaptiveAvgPool2d(1, 1)
	x := autograd.NewVar(tensor.RandN(4, 16, 7, 7), false)
	out := pool.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 4 || shape[1] != 16 || shape[2] != 1 || shape[3] != 1 {
		t.Fatalf("expected [4 16 1 1], got %v", shape)
	}
}

func TestAdaptiveAvgPool2dValues(t *testing.T) {
	// 1 sample, 1 channel, 2x2 input → 1x1 output should be mean
	pool := NewAdaptiveAvgPool2d(1, 1)
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4}, []int{1, 1, 2, 2}), false)
	out := pool.Forward(x)
	want := 2.5
	if math.Abs(out.Data.Data()[0]-want) > 1e-10 {
		t.Errorf("got %v want 2.5", out.Data.Data()[0])
	}
}

func TestAdaptiveAvgPool2dBackward(t *testing.T) {
	pool := NewAdaptiveAvgPool2d(2, 2)
	x := autograd.NewVar(tensor.RandN(2, 4, 8, 8), true)
	out := pool.Forward(x)
	loss := autograd.Mean(autograd.Sum(out))
	loss.Backward()
	if x.Grad == nil {
		t.Fatal("AdaptiveAvgPool2d: x.Grad is nil")
	}
}
