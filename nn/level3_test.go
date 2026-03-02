package nn

import (
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── Embedding ────────────────────────────────────────────────────────────────

func TestEmbeddingShape(t *testing.T) {
	emb := NewEmbedding(100, 32)
	out := emb.Lookup([]int{0, 5, 42})
	shape := out.Data.Shape()
	if shape[0] != 3 || shape[1] != 32 {
		t.Fatalf("expected [3 32], got %v", shape)
	}
}

func TestEmbeddingValues(t *testing.T) {
	emb := NewEmbedding(10, 4)
	// Force known weight row 3 = [1, 2, 3, 4]
	data := emb.Weight.Data.Data()
	data[3*4+0] = 1
	data[3*4+1] = 2
	data[3*4+2] = 3
	data[3*4+3] = 4
	emb.Weight.Data = tensor.New(data, []int{10, 4})

	out := emb.Lookup([]int{3})
	got := out.Data.Data()
	want := []float64{1, 2, 3, 4}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("index %d: got %v want %v", i, got[i], v)
		}
	}
}

func TestEmbeddingParameters(t *testing.T) {
	emb := NewEmbedding(50, 16)
	params := emb.Parameters()
	if len(params) != 1 {
		t.Fatalf("expected 1 parameter, got %d", len(params))
	}
	if params[0].Data.Shape()[0] != 50 || params[0].Data.Shape()[1] != 16 {
		t.Fatalf("wrong weight shape: %v", params[0].Data.Shape())
	}
}

// ── LSTM ──────────────────────────────────────────────────────────────────────

func TestLSTMShape(t *testing.T) {
	lstm := NewLSTM(8, 16)
	x := autograd.NewVar(tensor.RandN(5, 8), false) // [T=5, inputSize=8]
	outputs, state := lstm.Forward(x, nil)

	if len(outputs) != 5 {
		t.Fatalf("expected 5 outputs, got %d", len(outputs))
	}
	if outputs[0].Data.Shape()[0] != 16 {
		t.Fatalf("output[0] shape: %v, expected [16]", outputs[0].Data.Shape())
	}
	if state.H.Data.Shape()[0] != 16 {
		t.Fatalf("hN shape: %v, expected [16]", state.H.Data.Shape())
	}
}

func TestLSTMStateCarried(t *testing.T) {
	// Run two sequences and verify state is passed correctly
	lstm := NewLSTM(4, 8)
	x1 := autograd.NewVar(tensor.RandN(3, 4), false)
	_, state1 := lstm.Forward(x1, nil)

	x2 := autograd.NewVar(tensor.RandN(3, 4), false)
	_, state2 := lstm.Forward(x2, state1)

	// State should be non-zero
	hData := state2.H.Data.Data()
	nonzero := false
	for _, v := range hData {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Error("final state H should be non-zero")
	}
}

// ── GRU ───────────────────────────────────────────────────────────────────────

func TestGRUShape(t *testing.T) {
	gru := NewGRU(8, 16)
	x := autograd.NewVar(tensor.RandN(5, 8), false)
	outputs := gru.Forward(x, nil)

	if len(outputs) != 5 {
		t.Fatalf("expected 5 outputs, got %d", len(outputs))
	}
	if outputs[0].Data.Shape()[0] != 16 {
		t.Fatalf("output shape: %v, expected [16]", outputs[0].Data.Shape())
	}
}

func TestGRUParameters(t *testing.T) {
	gru := NewGRU(4, 8)
	params := gru.Parameters()
	if len(params) != 3 {
		t.Fatalf("expected 3 params (WX, WH, B), got %d", len(params))
	}
}

// ── MultiheadAttention ────────────────────────────────────────────────────────

func TestMHAShape(t *testing.T) {
	mha := NewMultiheadAttention(32, 4, true)
	x := autograd.NewVar(tensor.RandN(10, 32), false) // [T=10, embed=32]
	out := mha.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 10 || shape[1] != 32 {
		t.Fatalf("expected [10 32], got %v", shape)
	}
}

func TestMHAParameters(t *testing.T) {
	mha := NewMultiheadAttention(16, 2, true)
	params := mha.Parameters()
	// WQ, WK, WV, WO + BQ, BK, BV, BO = 8
	if len(params) != 8 {
		t.Fatalf("expected 8 params, got %d", len(params))
	}
}

func TestCausalMask(t *testing.T) {
	mask := CausalMask(4)
	// Upper triangle should be -1e9, lower+diagonal should be 0
	if mask.At(0, 1) != -1e9 {
		t.Errorf("(0,1) should be -1e9, got %v", mask.At(0, 1))
	}
	if mask.At(1, 0) != 0 {
		t.Errorf("(1,0) should be 0, got %v", mask.At(1, 0))
	}
	if mask.At(2, 2) != 0 {
		t.Errorf("diagonal should be 0, got %v", mask.At(2, 2))
	}
}

// ── TransformerEncoderLayer ──────────────────────────────────────────────────

func TestTransformerEncoderLayerShape(t *testing.T) {
	layer := NewTransformerEncoderLayer(32, 4, 64, 0.0)
	x := autograd.NewVar(tensor.RandN(6, 32), false) // [T=6, embed=32]
	out := layer.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 6 || shape[1] != 32 {
		t.Fatalf("expected [6 32], got %v", shape)
	}
}

func TestTransformerEncoderLayerParameters(t *testing.T) {
	layer := NewTransformerEncoderLayer(16, 2, 32, 0.0)
	params := layer.Parameters()
	// MHA(8) + FFN1(2) + FFN2(2) + Norm1(2) + Norm2(2) = 16
	if len(params) < 10 {
		t.Fatalf("expected at least 10 params, got %d", len(params))
	}
}
