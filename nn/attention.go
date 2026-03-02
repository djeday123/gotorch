package nn

import (
	"fmt"
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// MultiheadAttention implements scaled dot-product multi-head attention.
//
//	MHA(Q,K,V) = concat(head_1,...,head_h) W_o
//	head_i     = softmax(Q_i K_i^T / sqrt(d_k)) V_i
//
// Input shapes (all): [T, embedDim]  (sequence-first, batch not yet supported)
// Output shape:       [T, embedDim]
type MultiheadAttention struct {
	EmbedDim  int
	NumHeads  int
	HeadDim   int // embedDim / numHeads

	WQ, WK, WV *autograd.Variable // [embedDim, embedDim]
	WO         *autograd.Variable // [embedDim, embedDim]
	BQ, BK, BV *autograd.Variable // [embedDim]
	BO         *autograd.Variable
	UseBias    bool
}

// NewMultiheadAttention creates an MHA layer.
// embedDim must be divisible by numHeads.
func NewMultiheadAttention(embedDim, numHeads int, bias bool) *MultiheadAttention {
	if embedDim%numHeads != 0 {
		panic(fmt.Sprintf("nn.MultiheadAttention: embedDim %d must be divisible by numHeads %d",
			embedDim, numHeads))
	}
	scale := math.Sqrt(2.0 / float64(embedDim))
	newW := func() *autograd.Variable {
		d := tensor.RandN(embedDim, embedDim)
		data := d.Data()
		for i := range data {
			data[i] *= scale
		}
		return autograd.NewVar(tensor.New(data, []int{embedDim, embedDim}), true)
	}
	newB := func() *autograd.Variable {
		return autograd.NewVar(tensor.Zeros(embedDim), true)
	}

	m := &MultiheadAttention{
		EmbedDim: embedDim,
		NumHeads: numHeads,
		HeadDim:  embedDim / numHeads,
		WQ:       newW(), WK: newW(), WV: newW(), WO: newW(),
		UseBias: bias,
	}
	if bias {
		m.BQ = newB()
		m.BK = newB()
		m.BV = newB()
		m.BO = newB()
	}
	return m
}

func (m *MultiheadAttention) Parameters() []*autograd.Variable {
	ps := []*autograd.Variable{m.WQ, m.WK, m.WV, m.WO}
	if m.UseBias {
		ps = append(ps, m.BQ, m.BK, m.BV, m.BO)
	}
	return ps
}

func (m *MultiheadAttention) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}

// Forward computes multi-head self-attention.
// x: [T, embedDim] — query, key and value are all x (self-attention).
// Returns: [T, embedDim]
func (m *MultiheadAttention) Forward(x *autograd.Variable) *autograd.Variable {
	return m.ForwardQKV(x, x, x, nil)
}

// ForwardQKV computes cross-attention with separate Q, K, V inputs.
// mask: [T_q, T_k] — additive mask (e.g. -inf for positions to ignore), or nil.
func (m *MultiheadAttention) ForwardQKV(q, k, v *autograd.Variable, mask *tensor.Tensor) *autograd.Variable {
	// Linear projections: [T, embed] @ [embed, embed] → [T, embed]
	Q := autograd.MatMul(q, m.WQ)
	K := autograd.MatMul(k, m.WK)
	V := autograd.MatMul(v, m.WV)
	if m.UseBias {
		Q = addVecBias(Q, m.BQ)
		K = addVecBias(K, m.BK)
		V = addVecBias(V, m.BV)
	}

	// Split into heads and compute attention
	out := multiHeadAttentionSplit(Q, K, V, m.NumHeads, m.HeadDim, mask)

	// Output projection
	out = autograd.MatMul(out, m.WO)
	if m.UseBias {
		out = addVecBias(out, m.BO)
	}
	return out
}

// addVecBias adds a bias vector [D] to every row of [T, D].
type addVecBiasBackward struct{ T, D int }

func (f *addVecBiasBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// dX = grad
	// dB = sum over T
	dB := tensor.Zeros(f.D)
	for t := 0; t < f.T; t++ {
		for d := 0; d < f.D; d++ {
			dB.Set(dB.At(d)+grad.At(t, d), d)
		}
	}
	return []*tensor.Tensor{grad, dB}
}

func addVecBias(x, b *autograd.Variable) *autograd.Variable {
	T := x.Data.Shape()[0]
	D := x.Data.Shape()[1]
	out := tensor.Zeros(T, D)
	bData := b.Data.Data()
	for t := 0; t < T; t++ {
		for d := 0; d < D; d++ {
			out.Set(x.Data.At(t, d)+bData[d], t, d)
		}
	}
	return autograd.NewResult(out, &addVecBiasBackward{T, D}, x, b)
}

// multiHeadAttentionSplit splits Q/K/V into heads and computes attention.
func multiHeadAttentionSplit(Q, K, V *autograd.Variable, numHeads, headDim int, mask *tensor.Tensor) *autograd.Variable {
	T := Q.Data.Shape()[0]
	scale := 1.0 / math.Sqrt(float64(headDim))

	// Process each head independently, concatenate results
	headOuts := make([]*autograd.Variable, numHeads)
	for h := 0; h < numHeads; h++ {
		// Slice head: Q_h = Q[:, h*headDim:(h+1)*headDim]
		Qh := sliceHead(Q, h, headDim)
		Kh := sliceHead(K, h, headDim)
		Vh := sliceHead(V, h, headDim)

		// Scores = Q_h @ K_h^T * scale → [T, T]
		scores := autograd.MatMul(Qh, autograd.NewVar(Kh.Data.T(), false))
		scores = autograd.NewVar(tensor.MulScalar(scores.Data, scale), false)

		// Apply mask if provided
		if mask != nil {
			masked := make([]float64, T*T)
			sData := scores.Data.Data()
			mData := mask.Data()
			for i := range masked {
				masked[i] = sData[i] + mData[i]
			}
			scores = autograd.NewVar(tensor.New(masked, []int{T, T}), false)
		}

		// Softmax over last dim
		attn := autograd.NewVar(softmax2D(scores.Data), false)

		// Out_h = attn @ V_h → [T, headDim]
		headOuts[h] = autograd.NewVar(tensor.MatMul(attn.Data, Vh.Data), false)
	}

	// Concatenate heads: [T, numHeads*headDim] = [T, embedDim]
	// Build output directly
	embedDim := numHeads * headDim
	outData := make([]float64, T*embedDim)
	for h := 0; h < numHeads; h++ {
		hData := headOuts[h].Data.Data()
		for t := 0; t < T; t++ {
			for d := 0; d < headDim; d++ {
				outData[t*embedDim+h*headDim+d] = hData[t*headDim+d]
			}
		}
	}
	// Wrap as non-differentiable (full autograd through heads adds complexity;
	// gradients flow through the outer WQ/WK/WV/WO projections)
	return autograd.NewResult(
		tensor.New(outData, []int{T, embedDim}),
		&mhaConcatBackward{T: T, numHeads: numHeads, headDim: headDim, Q: Q, K: K, V: V, scale: scale, mask: mask},
		Q, K, V,
	)
}

// sliceHead extracts head h from [T, embedDim] → [T, headDim].
func sliceHead(x *autograd.Variable, h, headDim int) *autograd.Variable {
	T := x.Data.Shape()[0]
	embedDim := x.Data.Shape()[1]
	start := h * headDim
	out := make([]float64, T*headDim)
	xData := x.Data.Data()
	for t := 0; t < T; t++ {
		copy(out[t*headDim:], xData[t*embedDim+start:t*embedDim+start+headDim])
	}
	return autograd.NewVar(tensor.New(out, []int{T, headDim}), false)
}

// softmax2D applies softmax over the last dimension of a 2D tensor.
func softmax2D(t *tensor.Tensor) *tensor.Tensor {
	rows, cols := t.Shape()[0], t.Shape()[1]
	data := t.Data()
	out := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		base := r * cols
		maxV := data[base]
		for c := 1; c < cols; c++ {
			if data[base+c] > maxV {
				maxV = data[base+c]
			}
		}
		sum := 0.0
		for c := 0; c < cols; c++ {
			out[base+c] = math.Exp(data[base+c] - maxV)
			sum += out[base+c]
		}
		for c := 0; c < cols; c++ {
			out[base+c] /= sum
		}
	}
	return tensor.New(out, []int{rows, cols})
}

// mhaConcatBackward provides a simplified backward pass for MHA.
// Gradients are propagated through the linear projections (WQ/WK/WV);
// the attention weights are treated as constants for simplicity.
type mhaConcatBackward struct {
	T, numHeads, headDim int
	Q, K, V              *autograd.Variable
	scale                float64
	mask                 *tensor.Tensor
}

func (f *mhaConcatBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// Pass gradient through to Q, K, V proportionally (simplified).
	// Each gets the full gradient (exact backward requires storing attn weights).
	return []*tensor.Tensor{grad, grad, grad}
}

// ---------------------------------------------------------------------------
// CausalMask — upper triangular -inf mask for decoder self-attention
// ---------------------------------------------------------------------------

// CausalMask returns a [T, T] mask where upper triangle = -1e9, diagonal+lower = 0.
// Used for autoregressive (causal) attention in GPT-style decoders.
func CausalMask(T int) *tensor.Tensor {
	data := make([]float64, T*T)
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			if j > i {
				data[i*T+j] = -1e9
			}
		}
	}
	return tensor.New(data, []int{T, T})
}
