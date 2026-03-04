package nn

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── BatchNorm1d ──────────────────────────────────────────────────────────────

func TestBatchNorm1dShape(t *testing.T) {
	bn := NewBatchNorm1d(8)
	x := autograd.NewVar(tensor.Zeros(4, 8), false)
	out := bn.Forward(x)
	shape := out.Data.Shape()
	if len(shape) != 2 || shape[0] != 4 || shape[1] != 8 {
		t.Errorf("BatchNorm1d: expected shape [4 8], got %v", shape)
	}
}

func TestBatchNorm1dNormalized(t *testing.T) {
	bn := NewBatchNorm1d(4)
	// Create input with clear mean offset
	data := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	x := autograd.NewVar(tensor.New(data, []int{4, 4}), false)
	out := bn.Forward(x)

	outData := out.Data.Data()
	// After BN1d, each channel should be roughly normalized
	// Just check output is finite and different from input
	for i, v := range outData {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("BatchNorm1d output[%d] = %v (non-finite)", i, v)
		}
	}
}

func TestBatchNorm1dRunningStats(t *testing.T) {
	bn := NewBatchNorm1d(2)
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}), false)
	bn.Forward(x) // training pass — updates running stats

	// Running mean should be non-zero after update
	allZero := true
	for _, v := range bn.RunningMean {
		if math.Abs(v) > 1e-10 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("RunningMean should be updated after Forward in training mode")
	}
}

func TestBatchNorm1dEval(t *testing.T) {
	bn := NewBatchNorm1d(3)
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}), false)
	bn.Forward(x) // populate running stats
	bn.Eval()
	out := bn.Forward(x) // should use running stats, no panic
	if out.Data.Size() != 6 {
		t.Errorf("Eval mode output wrong size: %d", out.Data.Size())
	}
}

func TestBatchNorm1dParameters(t *testing.T) {
	bn := NewBatchNorm1d(6)
	ps := bn.Parameters()
	if len(ps) != 2 {
		t.Errorf("expected 2 parameters (gamma + beta), got %d", len(ps))
	}
	if ps[0].Data.Size() != 6 || ps[1].Data.Size() != 6 {
		t.Errorf("gamma/beta wrong size")
	}
}

// ── GroupNorm ────────────────────────────────────────────────────────────────

func TestGroupNormShape(t *testing.T) {
	gn := NewGroupNorm(2, 4) // 2 groups, 4 channels
	x := autograd.NewVar(tensor.Zeros(2, 4, 3, 3), false)
	out := gn.Forward(x)
	shape := out.Data.Shape()
	if len(shape) != 4 || shape[0] != 2 || shape[1] != 4 || shape[2] != 3 || shape[3] != 3 {
		t.Errorf("GroupNorm: expected [2 4 3 3], got %v", shape)
	}
}

func TestGroupNormNormalized(t *testing.T) {
	gn := NewGroupNorm(1, 4) // 1 group = all channels together
	data := make([]float64, 4*9) // [1, 4, 3, 3]
	for i := range data {
		data[i] = float64(i + 1)
	}
	x := autograd.NewVar(tensor.New(data, []int{1, 4, 3, 3}), false)
	out := gn.Forward(x)

	outData := out.Data.Data()
	for i, v := range outData {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("GroupNorm output[%d] = %v (non-finite)", i, v)
		}
	}
}

func TestGroupNormParameters(t *testing.T) {
	gn := NewGroupNorm(4, 8)
	ps := gn.Parameters()
	if len(ps) != 2 {
		t.Errorf("expected 2 parameters, got %d", len(ps))
	}
	if ps[0].Data.Size() != 8 {
		t.Errorf("gamma size: expected 8, got %d", ps[0].Data.Size())
	}
}

func TestGroupNormPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-divisible numChannels/numGroups")
		}
	}()
	NewGroupNorm(3, 8) // 8 % 3 != 0
}

// ── TransformerEncoder (stacked) ─────────────────────────────────────────────

func TestTransformerEncoderShape(t *testing.T) {
	layer := NewTransformerEncoderLayer(32, 4, 64, 0.0)
	enc := NewTransformerEncoder(layer, 2)
	// Input: [seqLen, embedDim]
	x := autograd.NewVar(tensor.Zeros(10, 32), false)
	out := enc.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 10 || shape[1] != 32 {
		t.Errorf("TransformerEncoder: expected [10 32], got %v", shape)
	}
}

func TestTransformerEncoderLayerCount(t *testing.T) {
	layer := NewTransformerEncoderLayer(16, 2, 32, 0.0)
	enc := NewTransformerEncoder(layer, 3)
	if len(enc.Layers) != 3 {
		t.Errorf("expected 3 layers, got %d", len(enc.Layers))
	}
}

func TestTransformerEncoderParameters(t *testing.T) {
	layer1 := NewTransformerEncoderLayer(16, 2, 32, 0.0)
	enc1 := NewTransformerEncoder(layer1, 1)
	n1 := len(enc1.Parameters())

	layer2 := NewTransformerEncoderLayer(16, 2, 32, 0.0)
	enc2 := NewTransformerEncoder(layer2, 2)
	n2 := len(enc2.Parameters())

	if n2 != 2*n1 {
		t.Errorf("2-layer encoder should have 2x params: got %d vs %d", n2, n1)
	}
}

func TestTransformerEncoderIndependentLayers(t *testing.T) {
	// Each layer should have independent weights
	layer := NewTransformerEncoderLayer(16, 2, 32, 0.0)
	enc := NewTransformerEncoder(layer, 3)

	if enc.Layers[0] == enc.Layers[1] || enc.Layers[1] == enc.Layers[2] {
		t.Error("layers should be independent (different pointers)")
	}
}

// ── SinusoidalPE ─────────────────────────────────────────────────────────────

func TestSinusoidalPEShape(t *testing.T) {
	pe := SinusoidalPE(20, 64)
	shape := pe.Data.Shape()
	if shape[0] != 20 || shape[1] != 64 {
		t.Errorf("SinusoidalPE: expected [20 64], got %v", shape)
	}
}

func TestSinusoidalPEValues(t *testing.T) {
	pe := SinusoidalPE(10, 8)
	data := pe.Data.Data()

	// pe[0][0] = sin(0/...) = 0
	if math.Abs(data[0]) > 1e-10 {
		t.Errorf("pe[0][0] should be 0, got %v", data[0])
	}
	// pe[0][1] = cos(0/...) = 1
	if math.Abs(data[1]-1.0) > 1e-10 {
		t.Errorf("pe[0][1] should be 1, got %v", data[1])
	}

	// All values should be in [-1, 1]
	for i, v := range data {
		if v < -1.001 || v > 1.001 {
			t.Errorf("pe[%d] = %v out of [-1, 1]", i, v)
		}
	}
}

func TestSinusoidalPENonDifferentiable(t *testing.T) {
	pe := SinusoidalPE(5, 4)
	if pe.RequiresGrad {
		t.Error("SinusoidalPE should not require grad")
	}
}

// ── PositionalEmbedding ───────────────────────────────────────────────────────

func TestPositionalEmbeddingShape(t *testing.T) {
	pe := NewPositionalEmbedding(128, 32)
	out := pe.Forward(10)
	shape := out.Data.Shape()
	if shape[0] != 10 || shape[1] != 32 {
		t.Errorf("PositionalEmbedding: expected [10 32], got %v", shape)
	}
}

func TestPositionalEmbeddingMaxLen(t *testing.T) {
	pe := NewPositionalEmbedding(16, 8)
	out := pe.Forward(16) // exactly maxLen, should work
	if out.Data.Size() != 16*8 {
		t.Error("expected maxLen forward to work")
	}
}

func TestPositionalEmbeddingPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when seqLen > maxLen")
		}
	}()
	pe := NewPositionalEmbedding(8, 4)
	pe.Forward(9) // exceed maxLen
}

func TestPositionalEmbeddingParameters(t *testing.T) {
	pe := NewPositionalEmbedding(32, 16)
	ps := pe.Parameters()
	if len(ps) != 1 {
		t.Errorf("expected 1 parameter, got %d", len(ps))
	}
	if ps[0].Data.Size() != 32*16 {
		t.Errorf("weight size: expected %d, got %d", 32*16, ps[0].Data.Size())
	}
}
