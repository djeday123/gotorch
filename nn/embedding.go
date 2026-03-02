package nn

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// Embedding is a lookup table: integer indices → dense vectors.
// Weight shape: [numEmbeddings, embeddingDim].
// Equivalent to torch.nn.Embedding.
type Embedding struct {
	NumEmbeddings int
	EmbeddingDim  int
	Weight        *autograd.Variable
}

// NewEmbedding creates an Embedding with Normal(0, 1) initialisation.
func NewEmbedding(numEmbeddings, embeddingDim int) *Embedding {
	std := 1.0 / math.Sqrt(float64(embeddingDim))
	w := tensor.RandN(numEmbeddings, embeddingDim)
	// Scale to small values
	wData := w.Data()
	for i := range wData {
		wData[i] *= std
	}
	return &Embedding{
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  embeddingDim,
		Weight:        autograd.NewVar(tensor.New(wData, []int{numEmbeddings, embeddingDim}), true),
	}
}

func (e *Embedding) Parameters() []*autograd.Variable { return []*autograd.Variable{e.Weight} }
func (e *Embedding) ZeroGrad()                         { e.Weight.ZeroGrad() }

// Lookup returns embeddings for the given integer indices.
// indices: []int of length T
// Output: [T, embeddingDim]
func (e *Embedding) Lookup(indices []int) *autograd.Variable {
	return embeddingForward(e.Weight, indices)
}

// Forward is an alias for Lookup that accepts a Variable wrapping a 1D integer-index tensor.
// The tensor values are cast to int indices. Output: [T, embeddingDim].
func (e *Embedding) Forward(x *autograd.Variable) *autograd.Variable {
	flat := x.Data.Data()
	indices := make([]int, len(flat))
	for i, v := range flat {
		indices[i] = int(v)
	}
	return e.Lookup(indices)
}

// ---------------------------------------------------------------------------

type embeddingBackward struct {
	indices       []int
	numEmbeddings int
	embeddingDim  int
}

func (f *embeddingBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// dWeight: scatter grad rows back to weight rows
	dW := tensor.Zeros(f.numEmbeddings, f.embeddingDim)
	dWData := dW.Data()
	gradData := grad.Data()

	for i, idx := range f.indices {
		for j := 0; j < f.embeddingDim; j++ {
			dWData[idx*f.embeddingDim+j] += gradData[i*f.embeddingDim+j]
		}
	}
	return []*tensor.Tensor{tensor.New(dWData, []int{f.numEmbeddings, f.embeddingDim})}
}

func embeddingForward(weight *autograd.Variable, indices []int) *autograd.Variable {
	numE := weight.Data.Shape()[0]
	dim := weight.Data.Shape()[1]
	wData := weight.Data.Data()

	T := len(indices)
	outData := make([]float64, T*dim)
	for i, idx := range indices {
		if idx < 0 || idx >= numE {
			panic("nn.Embedding: index out of range")
		}
		copy(outData[i*dim:], wData[idx*dim:(idx+1)*dim])
	}
	out := tensor.New(outData, []int{T, dim})
	return autograd.NewResult(out, &embeddingBackward{
		indices: indices, numEmbeddings: numE, embeddingDim: dim,
	}, weight)
}
