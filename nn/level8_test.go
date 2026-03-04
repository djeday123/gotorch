package nn

import (
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── TransformerDecoderLayer ──────────────────────────────────────────────────

func TestTransformerDecoderLayerShape(t *testing.T) {
	layer := NewTransformerDecoderLayer(32, 4, 64, 0.0)
	tgt := autograd.NewVar(tensor.Zeros(5, 32), false)
	mem := autograd.NewVar(tensor.Zeros(10, 32), false)
	out := layer.Forward(tgt, mem, nil)
	shape := out.Data.Shape()
	if shape[0] != 5 || shape[1] != 32 {
		t.Errorf("DecoderLayer: expected [5 32], got %v", shape)
	}
}

func TestTransformerDecoderLayerWithMask(t *testing.T) {
	layer := NewTransformerDecoderLayer(16, 2, 32, 0.0)
	tgt := autograd.NewVar(tensor.Zeros(4, 16), false)
	mem := autograd.NewVar(tensor.Zeros(6, 16), false)
	mask := CausalMask(4)
	out := layer.Forward(tgt, mem, mask)
	shape := out.Data.Shape()
	if shape[0] != 4 || shape[1] != 16 {
		t.Errorf("DecoderLayer+mask: expected [4 16], got %v", shape)
	}
}

func TestTransformerDecoderLayerParameters(t *testing.T) {
	layer := NewTransformerDecoderLayer(32, 4, 64, 0.0)
	ps := layer.Parameters()
	// SelfAttn + CrossAttn + FFN1 + FFN2 + Norm1 + Norm2 + Norm3
	if len(ps) == 0 {
		t.Error("expected non-zero parameters")
	}
	// Should have more params than EncoderLayer (extra CrossAttn + Norm3)
	encLayer := NewTransformerEncoderLayer(32, 4, 64, 0.0)
	if len(ps) <= len(encLayer.Parameters()) {
		t.Errorf("decoder should have more params than encoder layer: %d vs %d", len(ps), len(encLayer.Parameters()))
	}
}

// ── TransformerDecoder ───────────────────────────────────────────────────────

func TestTransformerDecoderShape(t *testing.T) {
	layer := NewTransformerDecoderLayer(32, 4, 64, 0.0)
	dec := NewTransformerDecoder(layer, 3)
	tgt := autograd.NewVar(tensor.Zeros(5, 32), false)
	mem := autograd.NewVar(tensor.Zeros(10, 32), false)
	out := dec.Forward(tgt, mem, nil)
	shape := out.Data.Shape()
	if shape[0] != 5 || shape[1] != 32 {
		t.Errorf("TransformerDecoder: expected [5 32], got %v", shape)
	}
}

func TestTransformerDecoderLayerCount(t *testing.T) {
	layer := NewTransformerDecoderLayer(16, 2, 32, 0.0)
	dec := NewTransformerDecoder(layer, 4)
	if len(dec.Layers) != 4 {
		t.Errorf("expected 4 layers, got %d", len(dec.Layers))
	}
}

func TestTransformerDecoderIndependentLayers(t *testing.T) {
	layer := NewTransformerDecoderLayer(16, 2, 32, 0.0)
	dec := NewTransformerDecoder(layer, 3)
	if dec.Layers[0] == dec.Layers[1] || dec.Layers[1] == dec.Layers[2] {
		t.Error("decoder layers should be independent")
	}
}

// ── nn.Transformer (full encoder-decoder) ────────────────────────────────────

func TestTransformerShape(t *testing.T) {
	model := NewTransformer(32, 4, 2, 2, 64, 0.0)
	src := autograd.NewVar(tensor.Zeros(10, 32), false)
	tgt := autograd.NewVar(tensor.Zeros(5, 32), false)
	out := model.Forward(src, tgt, nil)
	shape := out.Data.Shape()
	if shape[0] != 5 || shape[1] != 32 {
		t.Errorf("Transformer: expected [5 32], got %v", shape)
	}
}

func TestTransformerWithCausalMask(t *testing.T) {
	model := NewTransformer(16, 2, 1, 1, 32, 0.0)
	src := autograd.NewVar(tensor.Zeros(8, 16), false)
	tgt := autograd.NewVar(tensor.Zeros(6, 16), false)
	mask := CausalMask(6)
	out := model.Forward(src, tgt, mask)
	if out.Data.Shape()[0] != 6 {
		t.Errorf("Transformer+mask: expected tgtLen=6, got %v", out.Data.Shape())
	}
}

func TestTransformerParameters(t *testing.T) {
	model := NewTransformer(32, 4, 2, 2, 64, 0.0)
	ps := model.Parameters()
	if len(ps) == 0 {
		t.Error("expected non-empty parameters")
	}
	encPs := model.Encoder.Parameters()
	decPs := model.Decoder.Parameters()
	if len(ps) != len(encPs)+len(decPs) {
		t.Errorf("total params = encoder + decoder: %d != %d+%d", len(ps), len(encPs), len(decPs))
	}
}

// ── ModuleDict ───────────────────────────────────────────────────────────────

func TestModuleDictLen(t *testing.T) {
	d := NewModuleDict(map[string]Module{
		"a": NewLinear(4, 4, false),
		"b": NewLinear(4, 4, false),
	})
	if d.Len() != 2 {
		t.Errorf("expected len=2, got %d", d.Len())
	}
}

func TestModuleDictGet(t *testing.T) {
	lin := NewLinear(4, 4, false)
	d := NewModuleDict(map[string]Module{"fc": lin})
	got := d.Get("fc")
	if got == nil {
		t.Error("Get should return the module")
	}
}

func TestModuleDictGetMissing(t *testing.T) {
	d := NewModuleDict(nil)
	if d.Get("missing") != nil {
		t.Error("Get missing key should return nil")
	}
}

func TestModuleDictAdd(t *testing.T) {
	d := NewModuleDict(nil)
	d.Add("fc1", NewLinear(4, 8, true))
	d.Add("fc2", NewLinear(8, 4, true))
	if d.Len() != 2 {
		t.Errorf("expected 2 modules, got %d", d.Len())
	}
	if len(d.Parameters()) == 0 {
		t.Error("expected non-zero parameters")
	}
}

func TestModuleDictParameters(t *testing.T) {
	d := NewModuleDict(map[string]Module{
		"fc": NewLinear(4, 4, true),
	})
	ps := d.Parameters()
	// Linear(4,4, bias=true) → 2 params (W + b)
	if len(ps) != 2 {
		t.Errorf("expected 2 params, got %d", len(ps))
	}
}

// ── StackedLSTM ──────────────────────────────────────────────────────────────

func TestStackedLSTMShape(t *testing.T) {
	lstm := NewStackedLSTM(8, 16, 2)
	x := autograd.NewVar(tensor.Zeros(5, 8), false)
	h, c, out := lstm.Forward(x, nil)
	outShape := out.Data.Shape()
	if outShape[0] != 5 || outShape[1] != 16 {
		t.Errorf("StackedLSTM output: expected [5 16], got %v", outShape)
	}
	if h.Data.Size() != 16 || c.Data.Size() != 16 {
		t.Errorf("StackedLSTM h/c size wrong: h=%d c=%d", h.Data.Size(), c.Data.Size())
	}
}

func TestStackedLSTMSingleLayer(t *testing.T) {
	// 1-layer StackedLSTM should match regular LSTM
	lstm := NewStackedLSTM(4, 8, 1)
	x := autograd.NewVar(tensor.Zeros(3, 4), false)
	_, _, out := lstm.Forward(x, nil)
	shape := out.Data.Shape()
	if shape[0] != 3 || shape[1] != 8 {
		t.Errorf("1-layer StackedLSTM: expected [3 8], got %v", shape)
	}
}

func TestStackedLSTMParameters(t *testing.T) {
	lstm1 := NewStackedLSTM(4, 8, 1)
	lstm2 := NewStackedLSTM(4, 8, 2)
	n1 := len(lstm1.Parameters())
	n2 := len(lstm2.Parameters())
	if n2 != 2*n1 {
		t.Errorf("2-layer should have 2x params: %d vs %d", n2, n1)
	}
}

// ── StackedGRU ───────────────────────────────────────────────────────────────

func TestStackedGRUShape(t *testing.T) {
	gru := NewStackedGRU(6, 12, 3)
	x := autograd.NewVar(tensor.Zeros(7, 6), false)
	h, out := gru.Forward(x, nil)
	outShape := out.Data.Shape()
	if outShape[0] != 7 || outShape[1] != 12 {
		t.Errorf("StackedGRU output: expected [7 12], got %v", outShape)
	}
	if h.Data.Size() != 12 {
		t.Errorf("StackedGRU h size: expected 12, got %d", h.Data.Size())
	}
}

func TestStackedGRUParameters(t *testing.T) {
	gru1 := NewStackedGRU(4, 8, 1)
	gru2 := NewStackedGRU(4, 8, 2)
	n1 := len(gru1.Parameters())
	n2 := len(gru2.Parameters())
	if n2 != 2*n1 {
		t.Errorf("2-layer GRU should have 2x params: %d vs %d", n2, n1)
	}
}
