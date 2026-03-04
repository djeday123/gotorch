package nn

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ---------------------------------------------------------------------------
// SinusoidalPE — fixed sinusoidal positional encoding (Vaswani et al. 2017)
// ---------------------------------------------------------------------------

// SinusoidalPE returns a [seqLen, embedDim] tensor where:
//
//	pe[pos][2i]   = sin(pos / 10000^(2i/d))
//	pe[pos][2i+1] = cos(pos / 10000^(2i/d))
//
// The result is returned as a non-differentiable Variable (no gradient).
func SinusoidalPE(seqLen, embedDim int) *autograd.Variable {
	data := make([]float64, seqLen*embedDim)
	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < embedDim/2; i++ {
			angle := float64(pos) / math.Pow(10000.0, float64(2*i)/float64(embedDim))
			data[pos*embedDim+2*i] = math.Sin(angle)
			if 2*i+1 < embedDim {
				data[pos*embedDim+2*i+1] = math.Cos(angle)
			}
		}
	}
	return autograd.NewVar(tensor.New(data, []int{seqLen, embedDim}), false)
}

// ---------------------------------------------------------------------------
// PositionalEmbedding — learned positional embedding table
// ---------------------------------------------------------------------------

// PositionalEmbedding is a learned table of shape [maxLen, embedDim].
// Similar to nn.Embedding but indexed by position (0..seqLen-1).
//
//	pe := nn.NewPositionalEmbedding(512, 64)
//	pos := pe.Forward(seqLen)  // [seqLen, 64]
type PositionalEmbedding struct {
	MaxLen   int
	EmbedDim int
	Weight   *autograd.Variable // [MaxLen, EmbedDim]
}

// NewPositionalEmbedding creates a learned positional embedding table.
// Weights are initialised from a standard normal scaled by 0.02 (GPT-style).
func NewPositionalEmbedding(maxLen, embedDim int) *PositionalEmbedding {
	// Normal init scaled by 0.02
	data := make([]float64, maxLen*embedDim)
	scale := 0.02
	// Use a simple deterministic seed-like pattern for reproducibility in tests.
	// (In production use rand.NormFloat64())
	for i := range data {
		// Simple alternating sign approximation for init
		sign := 1.0
		if i%2 == 1 {
			sign = -1.0
		}
		data[i] = sign * scale * math.Sin(float64(i+1))
	}
	return &PositionalEmbedding{
		MaxLen:   maxLen,
		EmbedDim: embedDim,
		Weight:   autograd.NewVar(tensor.New(data, []int{maxLen, embedDim}), true),
	}
}

// Forward returns the first seqLen rows of the weight table: [seqLen, embedDim].
func (p *PositionalEmbedding) Forward(seqLen int) *autograd.Variable {
	if seqLen > p.MaxLen {
		panic("PositionalEmbedding: seqLen exceeds maxLen")
	}
	rows := p.Weight.Data.Data()[:seqLen*p.EmbedDim]
	rowsCopy := make([]float64, len(rows))
	copy(rowsCopy, rows)
	return autograd.NewVar(tensor.New(rowsCopy, []int{seqLen, p.EmbedDim}), false)
}

// Parameters returns the weight table for optimisation.
func (p *PositionalEmbedding) Parameters() []*autograd.Variable {
	return []*autograd.Variable{p.Weight}
}

// ZeroGrad zeros the weight gradient.
func (p *PositionalEmbedding) ZeroGrad() { p.Weight.ZeroGrad() }
