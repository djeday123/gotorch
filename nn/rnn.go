package nn

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ---------------------------------------------------------------------------
// LSTM — Long Short-Term Memory
// ---------------------------------------------------------------------------

// LSTM processes sequences step-by-step.
//
// Input:  x [T, inputSize]
// Output: out [T, hiddenSize], (hN, cN) = final hidden and cell states
//
// Gates: forget (f), input (i), cell (g), output (o)
//   f_t = sigmoid(x_t W_f + h_{t-1} U_f + b_f)
//   i_t = sigmoid(x_t W_i + h_{t-1} U_i + b_i)
//   g_t = tanh   (x_t W_g + h_{t-1} U_g + b_g)
//   o_t = sigmoid(x_t W_o + h_{t-1} U_o + b_o)
//   c_t = f_t * c_{t-1} + i_t * g_t
//   h_t = o_t * tanh(c_t)
type LSTM struct {
	InputSize  int
	HiddenSize int
	// Weights: [inputSize, 4*hiddenSize] for x gates (ifgo order)
	WX *autograd.Variable
	// Recurrent weights: [hiddenSize, 4*hiddenSize]
	WH *autograd.Variable
	// Bias: [4*hiddenSize]
	B *autograd.Variable
}

// NewLSTM creates an LSTM cell with Xavier initialisation.
func NewLSTM(inputSize, hiddenSize int) *LSTM {
	scale := math.Sqrt(1.0 / float64(hiddenSize))
	randW := func(rows, cols int) *autograd.Variable {
		d := tensor.RandN(rows, cols)
		data := d.Data()
		for i := range data {
			data[i] *= scale
		}
		return autograd.NewVar(tensor.New(data, []int{rows, cols}), true)
	}
	return &LSTM{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		WX:         randW(inputSize, 4*hiddenSize),
		WH:         randW(hiddenSize, 4*hiddenSize),
		B:          autograd.NewVar(tensor.Zeros(4*hiddenSize), true),
	}
}

func (l *LSTM) Parameters() []*autograd.Variable {
	return []*autograd.Variable{l.WX, l.WH, l.B}
}
func (l *LSTM) ZeroGrad() {
	l.WX.ZeroGrad()
	l.WH.ZeroGrad()
	l.B.ZeroGrad()
}

// LSTMState holds hidden and cell state.
type LSTMState struct {
	H, C *autograd.Variable
}

// Forward runs the LSTM over sequence x [T, inputSize].
// Returns outputs [T, hiddenSize] and final state.
// Initial h and c are zero if state is nil.
func (l *LSTM) Forward(x *autograd.Variable, state *LSTMState) ([]*autograd.Variable, *LSTMState) {
	T := x.Data.Shape()[0]
	H := l.HiddenSize

	var h, c *autograd.Variable
	if state != nil {
		h, c = state.H, state.C
	} else {
		h = autograd.NewVar(tensor.Zeros(H), false)
		c = autograd.NewVar(tensor.Zeros(H), false)
	}

	outputs := make([]*autograd.Variable, T)
	bData := l.B.Data.Data()

	for t := 0; t < T; t++ {
		// Get x_t: [1, inputSize] → [inputSize]
		xt := autograd.NewVar(x.Data.Select(0, t), false)

		// gates = x_t @ WX + h @ WH + b → [4*H]
		xGates := matVec(xt.Data, l.WX.Data)  // [4H]
		hGates := matVec(h.Data, l.WH.Data)   // [4H]

		gateData := make([]float64, 4*H)
		for j := 0; j < 4*H; j++ {
			gateData[j] = xGates[j] + hGates[j] + bData[j]
		}

		// Slice gates (i, f, g, o order — matching PyTorch)
		iG := sigmoid1D(gateData[0*H : 1*H])
		fG := sigmoid1D(gateData[1*H : 2*H])
		gG := tanh1D(gateData[2*H : 3*H])
		oG := sigmoid1D(gateData[3*H : 4*H])

		// c_t = f * c_{t-1} + i * g
		cData := c.Data.Data()
		newCData := make([]float64, H)
		for j := 0; j < H; j++ {
			newCData[j] = fG[j]*cData[j] + iG[j]*gG[j]
		}
		c = autograd.NewVar(tensor.New(newCData, []int{H}), false)

		// h_t = o * tanh(c_t)
		newHData := make([]float64, H)
		tanhC := tanh1D(newCData)
		for j := 0; j < H; j++ {
			newHData[j] = oG[j] * tanhC[j]
		}
		h = autograd.NewVar(tensor.New(newHData, []int{H}), true)
		outputs[t] = h
	}

	return outputs, &LSTMState{H: h, C: c}
}

// ---------------------------------------------------------------------------
// GRU — Gated Recurrent Unit
// ---------------------------------------------------------------------------

// GRU processes sequences with reset and update gates.
//
//   z_t = sigmoid(x_t W_z + h_{t-1} U_z + b_z)
//   r_t = sigmoid(x_t W_r + h_{t-1} U_r + b_r)
//   n_t = tanh   (x_t W_n + (r_t * h_{t-1}) U_n + b_n)
//   h_t = (1 - z_t) * h_{t-1} + z_t * n_t
type GRU struct {
	InputSize  int
	HiddenSize int
	WX         *autograd.Variable // [inputSize, 3*hiddenSize]
	WH         *autograd.Variable // [hiddenSize, 3*hiddenSize]
	B          *autograd.Variable // [3*hiddenSize]
}

func NewGRU(inputSize, hiddenSize int) *GRU {
	scale := math.Sqrt(1.0 / float64(hiddenSize))
	randW := func(rows, cols int) *autograd.Variable {
		d := tensor.RandN(rows, cols)
		data := d.Data()
		for i := range data {
			data[i] *= scale
		}
		return autograd.NewVar(tensor.New(data, []int{rows, cols}), true)
	}
	return &GRU{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		WX:         randW(inputSize, 3*hiddenSize),
		WH:         randW(hiddenSize, 3*hiddenSize),
		B:          autograd.NewVar(tensor.Zeros(3*hiddenSize), true),
	}
}

func (g *GRU) Parameters() []*autograd.Variable {
	return []*autograd.Variable{g.WX, g.WH, g.B}
}
func (g *GRU) ZeroGrad() {
	g.WX.ZeroGrad()
	g.WH.ZeroGrad()
	g.B.ZeroGrad()
}

// Forward runs GRU over x [T, inputSize]. h0 is zero if nil.
func (g *GRU) Forward(x *autograd.Variable, h0 *autograd.Variable) []*autograd.Variable {
	T := x.Data.Shape()[0]
	H := g.HiddenSize

	var h *autograd.Variable
	if h0 != nil {
		h = h0
	} else {
		h = autograd.NewVar(tensor.Zeros(H), false)
	}

	bData := g.B.Data.Data()
	outputs := make([]*autograd.Variable, T)

	for t := 0; t < T; t++ {
		xt := x.Data.Select(0, t)
		xGates := matVec(xt, g.WX.Data) // [3H]
		hGates := matVec(h.Data, g.WH.Data) // [3H]

		zData := make([]float64, H)
		rData := make([]float64, H)
		nInputData := make([]float64, H)

		for j := 0; j < H; j++ {
			zData[j] = xGates[j] + hGates[j] + bData[j]
			rData[j] = xGates[H+j] + hGates[H+j] + bData[H+j]
			nInputData[j] = xGates[2*H+j] + bData[2*H+j]
		}

		zG := sigmoid1D(zData)
		rG := sigmoid1D(rData)

		// n_t = tanh(xn + r * (hGates for n gate))
		nData := make([]float64, H)
		for j := 0; j < H; j++ {
			nData[j] = nInputData[j] + rG[j]*hGates[2*H+j]
		}
		nG := tanh1D(nData)

		// h_t = (1-z)*h + z*n
		hData := h.Data.Data()
		newH := make([]float64, H)
		for j := 0; j < H; j++ {
			newH[j] = (1-zG[j])*hData[j] + zG[j]*nG[j]
		}
		h = autograd.NewVar(tensor.New(newH, []int{H}), true)
		outputs[t] = h
	}
	return outputs
}

// ---------------------------------------------------------------------------
// TransformerEncoderLayer
// ---------------------------------------------------------------------------

// TransformerEncoderLayer = MultiheadAttention + FFN + LayerNorm + Dropout.
// Follows the Pre-LN variant (LayerNorm before sublayer).
//
//	x = x + Dropout(MHA(LN(x)))
//	x = x + Dropout(FFN(LN(x)))
type TransformerEncoderLayer struct {
	SelfAttn  *MultiheadAttention
	FFN1      *Linear // [embedDim → ffnDim]
	FFN2      *Linear // [ffnDim → embedDim]
	Norm1     *LayerNorm
	Norm2     *LayerNorm
	Drop1     *Dropout
	Drop2     *Dropout
	EmbedDim  int
	FFNDim    int
}

// NewTransformerEncoderLayer creates a Pre-LN transformer encoder layer.
// embedDim: model dimension. numHeads: attention heads. ffnDim: FFN hidden dim.
// dropoutP: dropout probability.
func NewTransformerEncoderLayer(embedDim, numHeads, ffnDim int, dropoutP float64) *TransformerEncoderLayer {
	return &TransformerEncoderLayer{
		SelfAttn: NewMultiheadAttention(embedDim, numHeads, true),
		FFN1:     NewLinear(embedDim, ffnDim, true),
		FFN2:     NewLinear(ffnDim, embedDim, true),
		Norm1:    NewLayerNorm([]int{embedDim}),
		Norm2:    NewLayerNorm([]int{embedDim}),
		Drop1:    NewDropout(dropoutP),
		Drop2:    NewDropout(dropoutP),
		EmbedDim: embedDim,
		FFNDim:   ffnDim,
	}
}

func (t *TransformerEncoderLayer) Parameters() []*autograd.Variable {
	var ps []*autograd.Variable
	ps = append(ps, t.SelfAttn.Parameters()...)
	ps = append(ps, t.FFN1.Parameters()...)
	ps = append(ps, t.FFN2.Parameters()...)
	ps = append(ps, t.Norm1.Parameters()...)
	ps = append(ps, t.Norm2.Parameters()...)
	return ps
}

func (t *TransformerEncoderLayer) ZeroGrad() {
	for _, p := range t.Parameters() {
		p.ZeroGrad()
	}
}

// Forward applies the transformer encoder layer to x [T, embedDim].
func (t *TransformerEncoderLayer) Forward(x *autograd.Variable) *autograd.Variable {
	// Self-attention block (Pre-LN)
	norm1Out := t.Norm1.Forward(x)
	attnOut := t.SelfAttn.Forward(norm1Out)
	attnOut = t.Drop1.Forward(attnOut)
	x = residualAdd(x, attnOut)

	// FFN block (Pre-LN)
	norm2Out := t.Norm2.Forward(x)
	ffnOut := t.FFN1.Forward(norm2Out)
	ffnOut = autograd.ReLU(ffnOut)
	ffnOut = t.FFN2.Forward(ffnOut)
	ffnOut = t.Drop2.Forward(ffnOut)
	x = residualAdd(x, ffnOut)

	return x
}

// residualAdd adds two Variables elementwise.
type residualAddBackward struct{}

func (f *residualAddBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{grad, grad}
}

func residualAdd(a, b *autograd.Variable) *autograd.Variable {
	aData := a.Data.Data()
	bData := b.Data.Data()
	out := make([]float64, len(aData))
	for i := range out {
		out[i] = aData[i] + bData[i]
	}
	return autograd.NewResult(
		tensor.New(out, a.Data.Shape()),
		&residualAddBackward{},
		a, b,
	)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// matVec computes x (1D [N]) @ W ([N, M]) → 1D [M]
func matVec(x, W *tensor.Tensor) []float64 {
	N := W.Shape()[0]
	M := W.Shape()[1]
	xData := x.Data()
	wData := W.Data()
	out := make([]float64, M)
	for m := 0; m < M; m++ {
		s := 0.0
		for n := 0; n < N; n++ {
			s += xData[n] * wData[n*M+m]
		}
		out[m] = s
	}
	return out
}

func sigmoid1D(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return out
}

func tanh1D(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Tanh(v)
	}
	return out
}

// ---------------------------------------------------------------------------
// ForwardSequence helpers for StackedLSTM / StackedGRU
// ---------------------------------------------------------------------------

// ForwardSequence runs LSTM over x [T, inputSize] with explicit h0/c0.
// Returns (finalH, finalC, outputs [T, hiddenSize]).
func (l *LSTM) ForwardSequence(x, h0, c0 *autograd.Variable) (*autograd.Variable, *autograd.Variable, *autograd.Variable) {
	var state *LSTMState
	if h0 != nil {
		state = &LSTMState{H: h0, C: c0}
	}
	outputs, finalState := l.Forward(x, state)
	return finalState.H, finalState.C, stackOutputs(outputs, l.HiddenSize)
}

// ForwardSequence runs GRU over x [T, inputSize] with explicit h0.
// Returns (finalH, outputs [T, hiddenSize]).
func (g *GRU) ForwardSequence(x, h0 *autograd.Variable) (*autograd.Variable, *autograd.Variable) {
	outputs := g.Forward(x, h0)
	return outputs[len(outputs)-1], stackOutputs(outputs, g.HiddenSize)
}

// stackOutputs concatenates a [T] slice of [H] variables into [T, H].
func stackOutputs(outs []*autograd.Variable, hiddenSize int) *autograd.Variable {
	T := len(outs)
	data := make([]float64, T*hiddenSize)
	for t, o := range outs {
		od := o.Data.Data()
		copy(data[t*hiddenSize:], od[:hiddenSize])
	}
	return autograd.NewVar(tensor.New(data, []int{T, hiddenSize}), false)
}

// zerosTensor1D returns a 1D zeros tensor of size n.
func zerosTensor1D(n int) *tensor.Tensor {
	return tensor.Zeros(n)
}
