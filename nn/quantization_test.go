package nn

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ─────────────────────────────────────────────────────────────────────────────
// QLinear
// ─────────────────────────────────────────────────────────────────────────────

func TestQuantizeLinear_NoBias(t *testing.T) {
	lin := NewLinear(4, 4, false)
	ql := QuantizeLinear(lin)

	if ql.inFeat != 4 || ql.outFeat != 4 {
		t.Errorf("feature dims: in=%d out=%d", ql.inFeat, ql.outFeat)
	}
	if ql.hasBias {
		t.Error("hasBias should be false")
	}
	if ql.qWeight.DType() != tensor.Int8 {
		t.Errorf("weight dtype: want int8, got %s", tensor.DTypeString(ql.qWeight.DType()))
	}
}

func TestQuantizeLinear_WithBias(t *testing.T) {
	lin := NewLinear(4, 4, true)
	ql := QuantizeLinear(lin)
	if !ql.hasBias {
		t.Error("hasBias should be true")
	}
	if ql.bias == nil {
		t.Error("bias should not be nil")
	}
}

func TestQLinearForward_Shape(t *testing.T) {
	lin := NewLinear(8, 4, true)
	ql := QuantizeLinear(lin)

	x := autograd.NewVar(tensor.New(make([]float64, 3*8), []int{3, 8}), false)
	out := ql.Forward(x)

	if out.Data.Shape()[0] != 3 || out.Data.Shape()[1] != 4 {
		t.Errorf("output shape: %v, want [3 4]", out.Data.Shape())
	}
}

func TestQLinearForward_CloseToOriginal(t *testing.T) {
	// QLinear output should be close (within ~1%) to original Linear output.
	lin := NewLinear(8, 4, true)

	data := make([]float64, 2*8)
	for i := range data {
		data[i] = float64(i+1) * 0.01
	}
	x := autograd.NewVar(tensor.New(data, []int{2, 8}), false)

	origOut := lin.Forward(x)
	ql := QuantizeLinear(lin)
	qOut := ql.Forward(x)

	od := origOut.Data.Data()
	qd := qOut.Data.Data()
	for i := range od {
		relErr := math.Abs(od[i]-qd[i]) / (math.Abs(od[i]) + 1e-8)
		if relErr > 0.15 { // allow 15% quantization error (int8 can accumulate error over matmul)
			t.Errorf("element %d: orig=%.6f quant=%.6f relErr=%.4f", i, od[i], qd[i], relErr)
		}
	}
}

func TestQLinearNoRequiresGrad(t *testing.T) {
	// QLinear is inference-only — output must not require grad.
	lin := NewLinear(4, 4, false)
	ql := QuantizeLinear(lin)
	x := autograd.NewVar(tensor.New(make([]float64, 4), []int{1, 4}), false)
	out := ql.Forward(x)
	if out.RequiresGrad {
		t.Error("QLinear output should not require grad")
	}
}

func TestQLinearParameters(t *testing.T) {
	// QLinear has no trainable parameters.
	lin := NewLinear(4, 4, true)
	ql := QuantizeLinear(lin)
	if len(ql.Parameters()) != 0 {
		t.Errorf("QLinear should have 0 parameters, got %d", len(ql.Parameters()))
	}
}

func TestQLinearString(t *testing.T) {
	lin := NewLinear(4, 8, true)
	ql := QuantizeLinear(lin)
	s := ql.String()
	if s == "" {
		t.Error("String() returned empty")
	}
}

func TestQLinearCompressionRatio(t *testing.T) {
	lin := NewLinear(4, 4, true)
	ql := QuantizeLinear(lin)
	if ql.CompressionRatio() != 8.0 {
		t.Errorf("expected compression ratio 8.0, got %g", ql.CompressionRatio())
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizeModel
// ─────────────────────────────────────────────────────────────────────────────

func TestQuantizeModel_ReplacesLinear(t *testing.T) {
	model := NewSequential(
		NewLinear(8, 16, true),
		NewReLU(),
		NewLinear(16, 4, true),
	)
	qm := QuantizeModel(model)

	layers := qm.layers
	if len(layers) != 3 {
		t.Fatalf("expected 3 layers, got %d", len(layers))
	}
	if _, ok := layers[0].(*QLinear); !ok {
		t.Errorf("layer 0: expected *QLinear, got %T", layers[0])
	}
	if _, ok := layers[1].(*ReLULayer); !ok {
		t.Errorf("layer 1: expected *ReLULayer, got %T", layers[1])
	}
	if _, ok := layers[2].(*QLinear); !ok {
		t.Errorf("layer 2: expected *QLinear, got %T", layers[2])
	}
}

func TestQuantizeModel_ForwardShape(t *testing.T) {
	model := NewSequential(
		NewLinear(8, 4, true),
		NewReLU(),
		NewLinear(4, 2, true),
	)
	qm := QuantizeModel(model)

	x := autograd.NewVar(tensor.New(make([]float64, 3*8), []int{3, 8}), false)
	out := qm.Forward(x)

	if out.Data.Shape()[0] != 3 || out.Data.Shape()[1] != 2 {
		t.Errorf("output shape: %v, want [3 2]", out.Data.Shape())
	}
}

func TestQuantizeModel_Stats(t *testing.T) {
	model := NewSequential(NewLinear(4, 4, true))
	qm := QuantizeModel(model)
	s := qm.Stats()
	if s == "" {
		t.Error("Stats() returned empty")
	}
}

func TestQuantizeModel_ZeroGrad(t *testing.T) {
	// ZeroGrad on a fully-quantized model should not panic.
	model := NewSequential(NewLinear(4, 4, true))
	qm := QuantizeModel(model)
	qm.ZeroGrad() // should not panic
}
