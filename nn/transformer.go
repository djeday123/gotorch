package nn

import "github.com/djeday123/gotorch/autograd"

// ---------------------------------------------------------------------------
// TransformerEncoder — stack of N TransformerEncoderLayer layers
// ---------------------------------------------------------------------------

// TransformerEncoder stacks numLayers copies of the given
// TransformerEncoderLayer architecture (sharing the same hyperparameters
// but with independent weights).
//
//	layer := nn.NewTransformerEncoderLayer(64, 8, 256, 0.1)
//	enc   := nn.NewTransformerEncoder(layer, 6)
//	out   := enc.Forward(x)   // x: [seqLen, embedDim]
type TransformerEncoder struct {
	Layers []*TransformerEncoderLayer
}

// NewTransformerEncoder creates a TransformerEncoder with numLayers independent
// layers that share the same hyperparameters as the prototype layer.
func NewTransformerEncoder(proto *TransformerEncoderLayer, numLayers int) *TransformerEncoder {
	embedDim := proto.EmbedDim
	ffnDim := proto.FFNDim
	numHeads := proto.SelfAttn.NumHeads
	var dropoutP float64
	if proto.Drop1 != nil {
		dropoutP = proto.Drop1.P
	}

	layers := make([]*TransformerEncoderLayer, numLayers)
	layers[0] = proto
	for i := 1; i < numLayers; i++ {
		layers[i] = NewTransformerEncoderLayer(embedDim, numHeads, ffnDim, dropoutP)
	}
	return &TransformerEncoder{Layers: layers}
}

// Forward passes x through each encoder layer sequentially.
func (e *TransformerEncoder) Forward(x *autograd.Variable) *autograd.Variable {
	for _, layer := range e.Layers {
		x = layer.Forward(x)
	}
	return x
}

// Parameters returns all learnable parameters from all layers.
func (e *TransformerEncoder) Parameters() []*autograd.Variable {
	var ps []*autograd.Variable
	for _, layer := range e.Layers {
		ps = append(ps, layer.Parameters()...)
	}
	return ps
}

// ZeroGrad zeros gradients in all layers.
func (e *TransformerEncoder) ZeroGrad() {
	for _, layer := range e.Layers {
		layer.ZeroGrad()
	}
}
