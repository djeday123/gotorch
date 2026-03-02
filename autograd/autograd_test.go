package autograd

import (
	"github.com/djeday123/gotorch/tensor"
	"math"
	"testing"
)

const eps = 1e-5
const tol = 1e-3

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < tol
}

// --- Basic grad tests ---

func TestAddGrad(t *testing.T) {
	x := NewVar(tensor.New([]float64{1, 2, 3}, []int{3}), true)
	y := NewVar(tensor.New([]float64{4, 5, 6}, []int{3}), true)
	z := Sum(Add(x, y))
	z.Backward()

	// dz/dx = 1, dz/dy = 1
	for i := 0; i < 3; i++ {
		if x.Grad.At(i) != 1.0 {
			t.Fatalf("x.Grad[%d] = %f, want 1.0", i, x.Grad.At(i))
		}
		if y.Grad.At(i) != 1.0 {
			t.Fatalf("y.Grad[%d] = %f, want 1.0", i, y.Grad.At(i))
		}
	}
}

func TestSubGrad(t *testing.T) {
	x := NewVar(tensor.New([]float64{5, 6}, []int{2}), true)
	y := NewVar(tensor.New([]float64{1, 2}, []int{2}), true)
	z := Sum(Sub(x, y))
	z.Backward()
	// dz/dx = 1, dz/dy = -1
	for i := 0; i < 2; i++ {
		if x.Grad.At(i) != 1.0 {
			t.Fatalf("x.Grad[%d] = %f, want 1.0", i, x.Grad.At(i))
		}
		if y.Grad.At(i) != -1.0 {
			t.Fatalf("y.Grad[%d] = %f, want -1.0", i, y.Grad.At(i))
		}
	}
}

func TestMulGrad(t *testing.T) {
	x := NewVar(tensor.New([]float64{2, 3}, []int{2}), true)
	y := NewVar(tensor.New([]float64{4, 5}, []int{2}), true)
	z := Sum(Mul(x, y))
	z.Backward()
	// dz/dx = y, dz/dy = x
	if x.Grad.At(0) != 4.0 || x.Grad.At(1) != 5.0 {
		t.Fatalf("x.Grad = %v, want [4,5]", x.Grad.Data())
	}
	if y.Grad.At(0) != 2.0 || y.Grad.At(1) != 3.0 {
		t.Fatalf("y.Grad = %v, want [2,3]", y.Grad.Data())
	}
}

func TestMatMulGrad(t *testing.T) {
	// a: 2x3, b: 3x2 -> c: 2x2
	a := NewVar(tensor.Rand(2, 3), true)
	b := NewVar(tensor.Rand(3, 2), true)
	c := Sum(MatMul(a, b))
	c.Backward()

	if a.Grad == nil {
		t.Fatal("a.Grad is nil")
	}
	if b.Grad == nil {
		t.Fatal("b.Grad is nil")
	}
	// Shapes must match
	if a.Grad.Shape()[0] != 2 || a.Grad.Shape()[1] != 3 {
		t.Fatalf("a.Grad shape = %v, want [2,3]", a.Grad.Shape())
	}
	if b.Grad.Shape()[0] != 3 || b.Grad.Shape()[1] != 2 {
		t.Fatalf("b.Grad shape = %v, want [3,2]", b.Grad.Shape())
	}
}

func TestNegGrad(t *testing.T) {
	x := NewVar(tensor.New([]float64{1, 2, 3}, []int{3}), true)
	z := Sum(Neg(x))
	z.Backward()
	for i := 0; i < 3; i++ {
		if x.Grad.At(i) != -1.0 {
			t.Fatalf("Neg grad[%d] = %f, want -1.0", i, x.Grad.At(i))
		}
	}
}

func TestExpGrad(t *testing.T) {
	xData := tensor.New([]float64{0, 1, 2}, []int{3})
	x := NewVar(xData, true)
	z := Sum(Exp(x))
	z.Backward()

	// Analytical: d/dx exp(x) = exp(x)
	numGrad := NumericalGrad(func(t *tensor.Tensor) float64 {
		s := tensor.Sum(tensor.Exp(t), -999, false)
		return s.Item()
	}, xData, eps)

	if !AllClose(x.Grad, numGrad, tol) {
		t.Fatalf("Exp grad mismatch: got %v, want %v", x.Grad.Data(), numGrad.Data())
	}
}

func TestLogGrad(t *testing.T) {
	xData := tensor.New([]float64{1, 2, 3}, []int{3})
	x := NewVar(xData, true)
	z := Sum(Log(x))
	z.Backward()

	numGrad := NumericalGrad(func(t *tensor.Tensor) float64 {
		return tensor.Sum(tensor.Log(t), -999, false).Item()
	}, xData, eps)

	if !AllClose(x.Grad, numGrad, tol) {
		t.Fatalf("Log grad mismatch: got %v, want %v", x.Grad.Data(), numGrad.Data())
	}
}

func TestReLUGrad(t *testing.T) {
	x := NewVar(tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}), true)
	z := Sum(ReLU(x))
	z.Backward()
	expected := []float64{0, 0, 0, 1, 1}
	for i, e := range expected {
		if x.Grad.At(i) != e {
			t.Fatalf("ReLU grad[%d] = %f, want %f", i, x.Grad.At(i), e)
		}
	}
}

func TestSigmoidGrad(t *testing.T) {
	xData := tensor.New([]float64{-1, 0, 1, 2}, []int{4})
	x := NewVar(xData, true)
	z := Sum(Sigmoid(x))
	z.Backward()

	numGrad := NumericalGrad(func(t *tensor.Tensor) float64 {
		return tensor.Sum(tensor.Sigmoid(t), -999, false).Item()
	}, xData, eps)

	if !AllClose(x.Grad, numGrad, tol) {
		t.Fatalf("Sigmoid grad mismatch: got %v, want %v", x.Grad.Data(), numGrad.Data())
	}
}

func TestTanhGrad(t *testing.T) {
	xData := tensor.New([]float64{-1, 0, 1}, []int{3})
	x := NewVar(xData, true)
	z := Sum(Tanh(x))
	z.Backward()

	numGrad := NumericalGrad(func(t *tensor.Tensor) float64 {
		return tensor.Sum(tensor.Tanh(t), -999, false).Item()
	}, xData, eps)

	if !AllClose(x.Grad, numGrad, tol) {
		t.Fatalf("Tanh grad mismatch: got %v, want %v", x.Grad.Data(), numGrad.Data())
	}
}

func TestPowScalarGrad(t *testing.T) {
	xData := tensor.New([]float64{1, 2, 3}, []int{3})
	x := NewVar(xData, true)
	z := Sum(PowScalar(x, 3))
	z.Backward()

	numGrad := NumericalGrad(func(t *tensor.Tensor) float64 {
		return tensor.Sum(tensor.PowScalar(t, 3), -999, false).Item()
	}, xData, eps)

	if !AllClose(x.Grad, numGrad, tol) {
		t.Fatalf("PowScalar grad mismatch: got %v, want %v", x.Grad.Data(), numGrad.Data())
	}
}

func TestMeanGrad(t *testing.T) {
	x := NewVar(tensor.New([]float64{1, 2, 3, 4}, []int{4}), true)
	z := Mean(x)
	z.Backward()
	// d/dx mean(x) = 1/n
	for i := 0; i < 4; i++ {
		if !almostEqual(x.Grad.At(i), 0.25) {
			t.Fatalf("Mean grad[%d] = %f, want 0.25", i, x.Grad.At(i))
		}
	}
}

func TestChainRule(t *testing.T) {
	// z = sum(relu(x * 2 + 1))
	xData := tensor.New([]float64{-1, 0, 1, 2}, []int{4})
	x := NewVar(xData, true)
	z := Sum(ReLU(AddScalar(MulScalar(x, 2), 1)))
	z.Backward()

	numGrad := NumericalGrad(func(t *tensor.Tensor) float64 {
		tmp := tensor.AddScalar(tensor.MulScalar(t, 2), 1)
		return tensor.Sum(tensor.ReLU(tmp), -999, false).Item()
	}, xData, eps)

	if !AllClose(x.Grad, numGrad, tol) {
		t.Fatalf("Chain rule grad mismatch: got %v, want %v", x.Grad.Data(), numGrad.Data())
	}
}

func TestNoGradLeaf(t *testing.T) {
	x := NewVar(tensor.Ones(3), false)
	y := NewVar(tensor.Ones(3), true)
	z := Sum(Add(x, y))
	z.Backward()
	if x.Grad != nil {
		t.Fatal("x (no grad) should have nil Grad")
	}
	if y.Grad == nil {
		t.Fatal("y (requires grad) should have non-nil Grad")
	}
}

func TestMatMulGradNumerical(t *testing.T) {
	aData := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	bData := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2})

	a := NewVar(aData, true)
	b := NewVar(bData, true)
	c := Sum(MatMul(a, b))
	c.Backward()

	numGradA := NumericalGrad(func(t *tensor.Tensor) float64 {
		return tensor.Sum(tensor.MatMul(t, bData), -999, false).Item()
	}, aData, eps)

	numGradB := NumericalGrad(func(t *tensor.Tensor) float64 {
		return tensor.Sum(tensor.MatMul(aData, t), -999, false).Item()
	}, bData, eps)

	if !AllClose(a.Grad, numGradA, tol) {
		t.Fatalf("MatMul grad_a mismatch\ngot:  %v\nwant: %v", a.Grad.Data(), numGradA.Data())
	}
	if !AllClose(b.Grad, numGradB, tol) {
		t.Fatalf("MatMul grad_b mismatch\ngot:  %v\nwant: %v", b.Grad.Data(), numGradB.Data())
	}
}
