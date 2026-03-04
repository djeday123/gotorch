package nn

import (
	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ---------------------------------------------------------------------------
// TransformerDecoderLayer
// ---------------------------------------------------------------------------
//
// Implements the Pre-LN Transformer decoder layer:
//
//	x = x + Drop(SelfAttn(LN(x), causal_mask))        // masked self-attention
//	x = x + Drop(CrossAttn(LN(x), memory, memory))    // cross-attention
//	x = x + Drop(FFN(LN(x)))                           // feed-forward
//
// memory is the encoder output [srcLen, embedDim].
type TransformerDecoderLayer struct {
	SelfAttn  *MultiheadAttention // masked self-attention
	CrossAttn *MultiheadAttention // cross-attention over encoder output
	FFN1      *Linear
	FFN2      *Linear
	Norm1     *LayerNorm
	Norm2     *LayerNorm
	Norm3     *LayerNorm
	Drop1     *Dropout
	Drop2     *Dropout
	Drop3     *Dropout
	EmbedDim  int
	FFNDim    int
}

// NewTransformerDecoderLayer creates a Pre-LN decoder layer.
func NewTransformerDecoderLayer(embedDim, numHeads, ffnDim int, dropoutP float64) *TransformerDecoderLayer {
	return &TransformerDecoderLayer{
		SelfAttn:  NewMultiheadAttention(embedDim, numHeads, true),
		CrossAttn: NewMultiheadAttention(embedDim, numHeads, true),
		FFN1:      NewLinear(embedDim, ffnDim, true),
		FFN2:      NewLinear(ffnDim, embedDim, true),
		Norm1:     NewLayerNorm([]int{embedDim}),
		Norm2:     NewLayerNorm([]int{embedDim}),
		Norm3:     NewLayerNorm([]int{embedDim}),
		Drop1:     NewDropout(dropoutP),
		Drop2:     NewDropout(dropoutP),
		Drop3:     NewDropout(dropoutP),
		EmbedDim:  embedDim,
		FFNDim:    ffnDim,
	}
}

// Forward runs one decoder layer.
//   - tgt:    target sequence [tgtLen, embedDim]
//   - memory: encoder output  [srcLen, embedDim]
//   - tgtMask: causal mask [tgtLen, tgtLen] (additive, -inf for future), or nil
func (d *TransformerDecoderLayer) Forward(tgt, memory *autograd.Variable, tgtMask *tensor.Tensor) *autograd.Variable {
	// 1. Masked self-attention
	norm1Out := d.Norm1.Forward(tgt)
	selfOut := d.SelfAttn.ForwardQKV(norm1Out, norm1Out, norm1Out, tgtMask)
	selfOut = d.Drop1.Forward(selfOut)
	tgt = residualAdd(tgt, selfOut)

	// 2. Cross-attention (Q=tgt, K=V=memory)
	norm2Out := d.Norm2.Forward(tgt)
	crossOut := d.CrossAttn.ForwardQKV(norm2Out, memory, memory, nil)
	crossOut = d.Drop2.Forward(crossOut)
	tgt = residualAdd(tgt, crossOut)

	// 3. Feed-forward
	norm3Out := d.Norm3.Forward(tgt)
	ffnOut := d.FFN2.Forward(autograd.ReLU(d.FFN1.Forward(norm3Out)))
	ffnOut = d.Drop3.Forward(ffnOut)
	tgt = residualAdd(tgt, ffnOut)

	return tgt
}

// Parameters returns all learnable parameters.
func (d *TransformerDecoderLayer) Parameters() []*autograd.Variable {
	var ps []*autograd.Variable
	ps = append(ps, d.SelfAttn.Parameters()...)
	ps = append(ps, d.CrossAttn.Parameters()...)
	ps = append(ps, d.FFN1.Parameters()...)
	ps = append(ps, d.FFN2.Parameters()...)
	ps = append(ps, d.Norm1.Parameters()...)
	ps = append(ps, d.Norm2.Parameters()...)
	ps = append(ps, d.Norm3.Parameters()...)
	return ps
}

// ZeroGrad zeros all gradients.
func (d *TransformerDecoderLayer) ZeroGrad() {
	for _, p := range d.Parameters() {
		p.ZeroGrad()
	}
}

// ---------------------------------------------------------------------------
// TransformerDecoder — stacked decoder layers
// ---------------------------------------------------------------------------

// TransformerDecoder stacks numLayers decoder layers.
type TransformerDecoder struct {
	Layers []*TransformerDecoderLayer
}

// NewTransformerDecoder creates a stacked decoder.
func NewTransformerDecoder(proto *TransformerDecoderLayer, numLayers int) *TransformerDecoder {
	layers := make([]*TransformerDecoderLayer, numLayers)
	layers[0] = proto
	for i := 1; i < numLayers; i++ {
		layers[i] = NewTransformerDecoderLayer(proto.EmbedDim,
			proto.SelfAttn.NumHeads, proto.FFNDim,
			proto.Drop1.P)
	}
	return &TransformerDecoder{Layers: layers}
}

// Forward passes tgt through all decoder layers.
func (dec *TransformerDecoder) Forward(tgt, memory *autograd.Variable, tgtMask *tensor.Tensor) *autograd.Variable {
	for _, layer := range dec.Layers {
		tgt = layer.Forward(tgt, memory, tgtMask)
	}
	return tgt
}

// Parameters returns all learnable parameters.
func (dec *TransformerDecoder) Parameters() []*autograd.Variable {
	var ps []*autograd.Variable
	for _, l := range dec.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}

// ZeroGrad zeros all gradients.
func (dec *TransformerDecoder) ZeroGrad() {
	for _, l := range dec.Layers {
		l.ZeroGrad()
	}
}

// ---------------------------------------------------------------------------
// Transformer — full encoder-decoder model
// ---------------------------------------------------------------------------

// Transformer combines a TransformerEncoder and TransformerDecoder.
// Equivalent to torch.nn.Transformer.
//
//	t := nn.NewTransformer(512, 8, 6, 6, 2048, 0.1)
//	out := t.Forward(src, tgt, srcMask, tgtMask)
type Transformer struct {
	Encoder *TransformerEncoder
	Decoder *TransformerDecoder
}

// NewTransformer creates a full encoder-decoder Transformer.
//   - dModel:    embedding dimension
//   - nHead:     number of attention heads
//   - numEncoderLayers / numDecoderLayers: depth
//   - dimFFN:    feed-forward hidden dim
//   - dropout:   dropout probability
func NewTransformer(dModel, nHead, numEncoderLayers, numDecoderLayers, dimFFN int, dropout float64) *Transformer {
	encLayer := NewTransformerEncoderLayer(dModel, nHead, dimFFN, dropout)
	decLayer := NewTransformerDecoderLayer(dModel, nHead, dimFFN, dropout)
	return &Transformer{
		Encoder: NewTransformerEncoder(encLayer, numEncoderLayers),
		Decoder: NewTransformerDecoder(decLayer, numDecoderLayers),
	}
}

// Forward encodes src and decodes tgt.
//   - src: [srcLen, dModel]
//   - tgt: [tgtLen, dModel]
//   - tgtMask: causal mask [tgtLen, tgtLen], or nil
func (t *Transformer) Forward(src, tgt *autograd.Variable, tgtMask *tensor.Tensor) *autograd.Variable {
	memory := t.Encoder.Forward(src)
	return t.Decoder.Forward(tgt, memory, tgtMask)
}

// Parameters returns all parameters from encoder + decoder.
func (t *Transformer) Parameters() []*autograd.Variable {
	ps := t.Encoder.Parameters()
	ps = append(ps, t.Decoder.Parameters()...)
	return ps
}

// ZeroGrad zeros all gradients.
func (t *Transformer) ZeroGrad() {
	t.Encoder.ZeroGrad()
	t.Decoder.ZeroGrad()
}

// Note: CausalMask is defined in attention.go
