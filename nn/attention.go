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
	embedDim := numHeads * headDim
	scale := 1.0 / math.Sqrt(float64(headDim))

	attns := make([]*tensor.Tensor, numHeads) // [T, T] per head — saved for backward
	outData := make([]float64, T*embedDim)

	for h := 0; h < numHeads; h++ {
		Qh := sliceHeadRaw(Q.Data, h, headDim) // [T, headDim]
		Kh := sliceHeadRaw(K.Data, h, headDim)
		Vh := sliceHeadRaw(V.Data, h, headDim)

		// scores = Qh @ Kh^T * scale  -> [T, T]
		scores := tensor.MulScalar(tensor.MatMul(Qh, Kh.T()), scale)

		// apply mask
		if mask != nil {
			scores = tensor.Add(scores, mask)
		}

		// softmax over last dim
		attn := softmax2D(scores)
		attns[h] = attn

		// head_out = attn @ Vh -> [T, headDim]
		headOut := tensor.MatMul(attn, Vh)
		hData := headOut.Data()
		for t := 0; t < T; t++ {
			for d := 0; d < headDim; d++ {
				outData[t*embedDim+h*headDim+d] = hData[t*headDim+d]
			}
		}
	}

	return autograd.NewResult(
		tensor.New(outData, []int{T, embedDim}),
		&mhaConcatBackward{
			T: T, numHeads: numHeads, headDim: headDim,
			Q: Q.Data, K: K.Data, V: V.Data,
			attns: attns, scale: scale,
		},
		Q, K, V,
	)
}

// sliceHeadRaw extracts head h from [T, embedDim] → [T, headDim] (raw tensor).
func sliceHeadRaw(x *tensor.Tensor, h, headDim int) *tensor.Tensor {
	T := x.Shape()[0]
	embedDim := x.Shape()[1]
	start := h * headDim
	out := make([]float64, T*headDim)
	xData := x.Data()
	for t := 0; t < T; t++ {
		copy(out[t*headDim:], xData[t*embedDim+start:t*embedDim+start+headDim])
	}
	return tensor.New(out, []int{T, headDim})
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

// mhaConcatBackward provides the full backward pass for multi-head attention.
// Stores attention weights and Q/K/V slices captured during forward.
type mhaConcatBackward struct {
	T, numHeads, headDim int
	Q, K, V              *tensor.Tensor   // [T, embedDim] — input data
	attns                []*tensor.Tensor // [T, T] per head
	scale                float64
}

func (f *mhaConcatBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	embedDim := f.numHeads * f.headDim
	dQ := make([]float64, f.T*embedDim)
	dK := make([]float64, f.T*embedDim)
	dV := make([]float64, f.T*embedDim)

	gradFlat := grad.Data()

	for h := 0; h < f.numHeads; h++ {
		// Slice grad for this head: dHead [T, headDim]
		dHeadFlat := make([]float64, f.T*f.headDim)
		for t := 0; t < f.T; t++ {
			for d := 0; d < f.headDim; d++ {
				dHeadFlat[t*f.headDim+d] = gradFlat[t*embedDim+h*f.headDim+d]
			}
		}
		dHead := tensor.New(dHeadFlat, []int{f.T, f.headDim})

		Qh := sliceHeadRaw(f.Q, h, f.headDim)
		Kh := sliceHeadRaw(f.K, h, f.headDim)
		Vh := sliceHeadRaw(f.V, h, f.headDim)
		attn := f.attns[h] // [T, T]

		// head_out = attn @ Vh
		// dVh    = attn^T @ dHead          -> [T, headDim]
		// dAttn  = dHead @ Vh^T            -> [T, T]
		dVh := tensor.MatMul(attn.T(), dHead)
		dAttn := tensor.MatMul(dHead, Vh.T())

		// softmax backward (row-wise): dS_i = s_i * (g_i - sum_j(g_j*s_j))
		dScores := softmaxBackward2D(dAttn, attn)

		// scores = Qh @ Kh^T * scale
		// dQh = dScores @ Kh * scale       -> [T, headDim]
		// dKh = dScores^T @ Qh * scale     -> [T, headDim]
		dQh := tensor.MulScalar(tensor.MatMul(dScores, Kh), f.scale)
		dKh := tensor.MulScalar(tensor.MatMul(dScores.T(), Qh), f.scale)

		// Write head gradients back into dQ / dK / dV
		dQhData := dQh.Data()
		dKhData := dKh.Data()
		dVhData := dVh.Data()
		for t := 0; t < f.T; t++ {
			for d := 0; d < f.headDim; d++ {
				idx := t*embedDim + h*f.headDim + d
				dQ[idx] = dQhData[t*f.headDim+d]
				dK[idx] = dKhData[t*f.headDim+d]
				dV[idx] = dVhData[t*f.headDim+d]
			}
		}
	}

	shape := []int{f.T, embedDim}
	return []*tensor.Tensor{
		tensor.New(dQ, shape),
		tensor.New(dK, shape),
		tensor.New(dV, shape),
	}
}

// softmaxBackward2D computes gradient through row-wise softmax of a [R, C] tensor.
// Given upstream g [R, C] and softmax output s [R, C], returns dS [R, C] where
// dS_i = s_i * (g_i - sum_j(g_j * s_j)) per row.
func softmaxBackward2D(g, s *tensor.Tensor) *tensor.Tensor {
	rows, cols := s.Shape()[0], s.Shape()[1]
	gData := g.Data()
	sData := s.Data()
	out := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		base := r * cols
		dot := 0.0
		for c := 0; c < cols; c++ {
			dot += gData[base+c] * sData[base+c]
		}
		for c := 0; c < cols; c++ {
			out[base+c] = sData[base+c] * (gData[base+c] - dot)
		}
	}
	return tensor.New(out, []int{rows, cols})
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
