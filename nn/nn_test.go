package nn

import (
	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/optim"
	"github.com/djeday123/gotorch/tensor"
	"math"
	"testing"
)

// ---- Linear ----

func TestLinearForwardShape(t *testing.T) {
	l := NewLinear(4, 8, true)
	x := autograd.NewVar(tensor.Rand(16, 4), false)
	out := l.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 16 || shape[1] != 8 {
		t.Fatalf("Linear output shape = %v, want [16,8]", shape)
	}
}

func TestLinearParameters(t *testing.T) {
	l := NewLinear(3, 5, true)
	params := l.Parameters()
	if len(params) != 2 {
		t.Fatalf("expected 2 params (W,b), got %d", len(params))
	}
	wShape := params[0].Data.Shape()
	if wShape[0] != 5 || wShape[1] != 3 {
		t.Fatalf("Weight shape = %v, want [5,3]", wShape)
	}
}

func TestLinearNoBias(t *testing.T) {
	l := NewLinear(3, 5, false)
	if len(l.Parameters()) != 1 {
		t.Fatal("no-bias linear should have 1 parameter")
	}
}

func TestLinearGrad(t *testing.T) {
	l := NewLinear(4, 3, true)
	x := autograd.NewVar(tensor.Rand(2, 4), false)
	out := l.Forward(x)
	loss := autograd.Sum(autograd.Mean(out))
	loss.Backward()

	if l.Weight.Grad == nil {
		t.Fatal("Weight.Grad is nil after backward")
	}
	if l.Bias.Grad == nil {
		t.Fatal("Bias.Grad is nil after backward")
	}
	// Weight grad shape should be (3, 4)
	ws := l.Weight.Grad.Shape()
	if ws[0] != 3 || ws[1] != 4 {
		t.Fatalf("Weight.Grad shape = %v, want [3,4]", ws)
	}
}

func TestLinearZeroGrad(t *testing.T) {
	l := NewLinear(2, 2, true)
	x := autograd.NewVar(tensor.Rand(1, 2), false)
	loss := autograd.Sum(l.Forward(x))
	loss.Backward()
	l.ZeroGrad()
	if l.Weight.Grad != nil {
		t.Fatal("ZeroGrad should nil Weight.Grad")
	}
}

// ---- Sequential ----

func TestSequentialForward(t *testing.T) {
	model := NewSequential(
		NewLinear(4, 8, true),
		NewReLU(),
		NewLinear(8, 2, true),
	)
	x := autograd.NewVar(tensor.Rand(5, 4), false)
	out := model.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 5 || shape[1] != 2 {
		t.Fatalf("Sequential output shape = %v, want [5,2]", shape)
	}
}

func TestSequentialParameters(t *testing.T) {
	model := NewSequential(
		NewLinear(4, 8, true),
		NewReLU(),
		NewLinear(8, 3, true),
	)
	// L1: 2 params, L2: 2 params = 4 total
	params := model.Parameters()
	if len(params) != 4 {
		t.Fatalf("expected 4 params, got %d", len(params))
	}
}

// ---- Loss ----

func TestMSELoss(t *testing.T) {
	pred := autograd.NewVar(tensor.New([]float64{1, 2, 3}, []int{3}), false)
	target := autograd.NewVar(tensor.New([]float64{1, 2, 3}, []int{3}), false)
	loss := MSELoss(pred, target)
	if math.Abs(loss.Data.Item()) > 1e-9 {
		t.Fatalf("MSELoss on equal tensors = %f, want 0.0", loss.Data.Item())
	}
}

func TestMSELossValue(t *testing.T) {
	pred := autograd.NewVar(tensor.New([]float64{0, 0}, []int{2}), false)
	target := autograd.NewVar(tensor.New([]float64{1, 1}, []int{2}), false)
	loss := MSELoss(pred, target)
	// MSE = mean((0-1)^2, (0-1)^2) = 1.0
	if math.Abs(loss.Data.Item()-1.0) > 1e-9 {
		t.Fatalf("MSELoss = %f, want 1.0", loss.Data.Item())
	}
}

func TestBCELoss(t *testing.T) {
	// Confident correct predictions → low loss
	pred := autograd.NewVar(tensor.New([]float64{0.99, 0.01}, []int{2}), false)
	target := autograd.NewVar(tensor.New([]float64{1.0, 0.0}, []int{2}), false)
	loss := BCELoss(pred, target)
	if loss.Data.Item() > 0.1 {
		t.Fatalf("BCELoss on confident predictions = %f, want < 0.1", loss.Data.Item())
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	// Logits heavily favour class 0 for sample 0, class 1 for sample 1
	logits := autograd.NewVar(tensor.New([]float64{10, 0, 0, 10}, []int{2, 2}), false)
	targets := []int{0, 1}
	loss := CrossEntropyLoss(logits, targets)
	if loss.Data.Item() > 0.01 {
		t.Fatalf("CrossEntropyLoss on confident logits = %f, want < 0.01", loss.Data.Item())
	}
}

// ---- XOR integration test ----
// Train a 2-layer MLP (2→4→1) on the XOR problem.
// After 3000 steps with Adam, predictions should be almost correct.

func TestXOR(t *testing.T) {
	// XOR truth table
	xData := tensor.New([]float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}, []int{4, 2})
	yData := tensor.New([]float64{0, 1, 1, 0}, []int{4, 1})

	model := NewSequential(
		NewLinear(2, 8, true),
		NewTanh(),
		NewLinear(8, 1, true),
		NewSigmoid(),
	)
	optimizer := optim.NewAdam(model.Parameters(), 0.05, 0.9, 0.999, 1e-8)

	x := autograd.NewVar(xData, false)
	y := autograd.NewVar(yData, false)

	var finalLoss float64
	for step := 0; step < 3000; step++ {
		optimizer.ZeroGrad()
		pred := model.Forward(x)
		loss := MSELoss(pred, y)
		loss.Backward()
		optimizer.Step()
		finalLoss = loss.Data.Item()
	}

	if finalLoss > 0.01 {
		t.Fatalf("XOR training failed: final loss = %f (want < 0.01)", finalLoss)
	}

	// Verify predictions
	pred := model.Forward(x).Data.Data()
	correct := 0
	expected := []float64{0, 1, 1, 0}
	for i, p := range pred {
		rounded := 0.0
		if p > 0.5 {
			rounded = 1.0
		}
		if rounded == expected[i] {
			correct++
		}
	}
	if correct < 4 {
		t.Fatalf("XOR predictions wrong: got %d/4 correct. Preds: %v", correct, pred)
	}
}

// ---- Activation modules ----

func TestActivationModules(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{-1, 0, 1}, []int{3}), false)

	relu := NewReLU()
	out := relu.Forward(x)
	if out.Data.At(0) != 0 || out.Data.At(2) != 1 {
		t.Fatal("ReLU module incorrect")
	}

	sig := NewSigmoid()
	out2 := sig.Forward(x)
	if out2.Data.At(1) != 0.5 {
		t.Fatalf("Sigmoid(0) = %f, want 0.5", out2.Data.At(1))
	}

	tanh := NewTanh()
	out3 := tanh.Forward(x)
	if math.Abs(out3.Data.At(1)) > 1e-9 {
		t.Fatalf("Tanh(0) = %f, want 0.0", out3.Data.At(1))
	}
}
