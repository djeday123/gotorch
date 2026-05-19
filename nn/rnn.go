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
//
// Implemented as a single autograd op over the whole sequence so gradients
// flow correctly through every gate at every timestep (full BPTT).
func (l *LSTM) Forward(x *autograd.Variable, state *LSTMState) ([]*autograd.Variable, *LSTMState) {
	T := x.Data.Shape()[0]
	H := l.HiddenSize

	var h0, c0 *autograd.Variable
	if state != nil {
		h0, c0 = state.H, state.C
	} else {
		h0 = autograd.NewVar(tensor.Zeros(H), false)
		c0 = autograd.NewVar(tensor.Zeros(H), false)
	}

	stackedH, finalC, _ := lstmSequenceForward(x, h0, c0, l.WX, l.WH, l.B)

	// Expose per-timestep outputs as autograd-aware row slices of stackedH.
	outputs := make([]*autograd.Variable, T)
	for t := 0; t < T; t++ {
		outputs[t] = rowSlice(stackedH, t)
	}

	return outputs, &LSTMState{H: outputs[T-1], C: finalC}
}

// lstmSequenceForward runs the LSTM cell over the full sequence and registers
// a single autograd op that implements BPTT in its backward pass.
//
// Returns:
//   stackedH [T, H]       — differentiable
//   finalC   [H]          — non-differentiable convenience handle
//   intermediates         — used by tests
func lstmSequenceForward(x, h0, c0, WX, WH, B *autograd.Variable) (
	*autograd.Variable, *autograd.Variable, *lstmIntermediates,
) {
	T := x.Data.Shape()[0]
	I := x.Data.Shape()[1]
	H := h0.Data.Size()

	xData := x.Data.Data()
	wxData := WX.Data.Data()
	whData := WH.Data.Data()
	bData := B.Data.Data()

	im := &lstmIntermediates{
		T: T, I: I, H: H,
		xt:     make([][]float64, T),
		hPrev:  make([][]float64, T),
		cPrev:  make([][]float64, T),
		i:      make([][]float64, T),
		f:      make([][]float64, T),
		g:      make([][]float64, T),
		o:      make([][]float64, T),
		tanhC:  make([][]float64, T),
	}

	hPrev := append([]float64(nil), h0.Data.Data()...)
	cPrev := append([]float64(nil), c0.Data.Data()...)

	stackedHData := make([]float64, T*H)

	for t := 0; t < T; t++ {
		xt := xData[t*I : (t+1)*I]
		im.xt[t] = append([]float64(nil), xt...)
		im.hPrev[t] = append([]float64(nil), hPrev...)
		im.cPrev[t] = append([]float64(nil), cPrev...)

		// linGates = xt @ WX + hPrev @ WH + b  -> [4H]
		linGates := make([]float64, 4*H)
		for m := 0; m < 4*H; m++ {
			s := bData[m]
			for n := 0; n < I; n++ {
				s += xt[n] * wxData[n*4*H+m]
			}
			for n := 0; n < H; n++ {
				s += hPrev[n] * whData[n*4*H+m]
			}
			linGates[m] = s
		}

		iG := make([]float64, H)
		fG := make([]float64, H)
		gG := make([]float64, H)
		oG := make([]float64, H)
		for j := 0; j < H; j++ {
			iG[j] = sigmoidScalar(linGates[0*H+j])
			fG[j] = sigmoidScalar(linGates[1*H+j])
			gG[j] = math.Tanh(linGates[2*H+j])
			oG[j] = sigmoidScalar(linGates[3*H+j])
		}
		im.i[t] = iG
		im.f[t] = fG
		im.g[t] = gG
		im.o[t] = oG

		cNew := make([]float64, H)
		hNew := make([]float64, H)
		tanhC := make([]float64, H)
		for j := 0; j < H; j++ {
			cNew[j] = fG[j]*cPrev[j] + iG[j]*gG[j]
			tanhC[j] = math.Tanh(cNew[j])
			hNew[j] = oG[j] * tanhC[j]
		}
		im.tanhC[t] = tanhC

		copy(stackedHData[t*H:(t+1)*H], hNew)
		cPrev = cNew
		hPrev = hNew
	}

	stackedH := autograd.NewResult(
		tensor.New(stackedHData, []int{T, H}),
		&lstmSequenceBackward{im: im, wxData: wxData, whData: whData},
		x, h0, c0, WX, WH, B,
	)
	finalC := autograd.NewVar(tensor.New(cPrev, []int{H}), false)
	return stackedH, finalC, im
}

type lstmIntermediates struct {
	T, I, H int
	xt      [][]float64
	hPrev   [][]float64
	cPrev   [][]float64
	i, f, g [][]float64
	o       [][]float64
	tanhC   [][]float64
}

type lstmSequenceBackward struct {
	im     *lstmIntermediates
	wxData []float64 // [I, 4H]
	whData []float64 // [H, 4H]
}

// Apply implements BPTT for an entire LSTM sequence.
// Inputs in order: x [T,I], h0 [H], c0 [H], WX [I,4H], WH [H,4H], B [4H].
// grad is upstream gradient on stackedH [T, H].
func (b *lstmSequenceBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	im := b.im
	T, I, H := im.T, im.I, im.H
	gradFlat := grad.Data()
	wxRef := b.wxData
	whRef := b.whData

	dX := make([]float64, T*I)
	dWX := make([]float64, I*4*H)
	dWH := make([]float64, H*4*H)
	dB := make([]float64, 4*H)
	dhNext := make([]float64, H)
	dcNext := make([]float64, H)

	for t := T - 1; t >= 0; t-- {
		i := im.i[t]
		f := im.f[t]
		g := im.g[t]
		o := im.o[t]
		tc := im.tanhC[t]
		cPrev := im.cPrev[t]
		hPrev := im.hPrev[t]
		xt := im.xt[t]

		// Upstream gradient on h_t: from stacked output + from next step's h_{t-1} = h_t
		dh := make([]float64, H)
		for j := 0; j < H; j++ {
			dh[j] = gradFlat[t*H+j] + dhNext[j]
		}
		// dc_t: only from next step (c_t isn't an output of stackedH)
		dc := make([]float64, H)
		copy(dc, dcNext)

		// h_t = o * tanh(c_t)  →  do = dh * tanh_c,  d_tanh_c = dh * o
		// c_t flow: dc += d_tanh_c * (1 - tanh_c^2)
		dLinGates := make([]float64, 4*H)
		for j := 0; j < H; j++ {
			doVal := dh[j] * tc[j]
			dTanhC := dh[j] * o[j]
			dc[j] += dTanhC * (1 - tc[j]*tc[j])

			// c_t = f*cPrev + i*g
			df := dc[j] * cPrev[j]
			di := dc[j] * g[j]
			dg := dc[j] * i[j]
			dcPrev := dc[j] * f[j]

			// sigmoid'(x) = s*(1-s); tanh'(x) = 1 - t^2
			dLinGates[0*H+j] = di * i[j] * (1 - i[j])
			dLinGates[1*H+j] = df * f[j] * (1 - f[j])
			dLinGates[2*H+j] = dg * (1 - g[j]*g[j])
			dLinGates[3*H+j] = doVal * o[j] * (1 - o[j])

			dcNext[j] = dcPrev
		}

		// db += dLinGates
		for m := 0; m < 4*H; m++ {
			dB[m] += dLinGates[m]
		}

		// dxt = dLinGates @ WX^T;  dWX += xt^T @ dLinGates (outer product)
		for n := 0; n < I; n++ {
			s := 0.0
			for m := 0; m < 4*H; m++ {
				s += dLinGates[m] * wxRef[n*4*H+m]
			}
			dX[t*I+n] = s
		}
		for n := 0; n < I; n++ {
			for m := 0; m < 4*H; m++ {
				dWX[n*4*H+m] += xt[n] * dLinGates[m]
			}
		}

		// dh_{t-1} = dLinGates @ WH^T;  dWH += hPrev^T @ dLinGates
		for n := 0; n < H; n++ {
			s := 0.0
			for m := 0; m < 4*H; m++ {
				s += dLinGates[m] * whRef[n*4*H+m]
			}
			dhNext[n] = s
		}
		for n := 0; n < H; n++ {
			for m := 0; m < 4*H; m++ {
				dWH[n*4*H+m] += hPrev[n] * dLinGates[m]
			}
		}
	}

	dh0 := dhNext
	dc0 := dcNext

	return []*tensor.Tensor{
		tensor.New(dX, []int{T, I}),
		tensor.New(dh0, []int{H}),
		tensor.New(dc0, []int{H}),
		tensor.New(dWX, []int{I, 4 * H}),
		tensor.New(dWH, []int{H, 4 * H}),
		tensor.New(dB, []int{4 * H}),
	}
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
//
// Implemented as a single autograd op with full BPTT through reset/update gates.
func (g *GRU) Forward(x *autograd.Variable, h0 *autograd.Variable) []*autograd.Variable {
	T := x.Data.Shape()[0]
	H := g.HiddenSize

	if h0 == nil {
		h0 = autograd.NewVar(tensor.Zeros(H), false)
	}

	stacked := gruSequenceForward(x, h0, g.WX, g.WH, g.B)
	outputs := make([]*autograd.Variable, T)
	for t := 0; t < T; t++ {
		outputs[t] = rowSlice(stacked, t)
	}
	return outputs
}

func gruSequenceForward(x, h0, WX, WH, B *autograd.Variable) *autograd.Variable {
	T := x.Data.Shape()[0]
	I := x.Data.Shape()[1]
	H := h0.Data.Size()

	xData := x.Data.Data()
	wxData := WX.Data.Data()
	whData := WH.Data.Data()
	bData := B.Data.Data()

	im := &gruIntermediates{
		T: T, I: I, H: H,
		xt:     make([][]float64, T),
		hPrev:  make([][]float64, T),
		z:      make([][]float64, T),
		r:      make([][]float64, T),
		n:      make([][]float64, T),
		hnInput: make([][]float64, T), // hPrev @ WH for n-slice (hGates_n) — needed for dr
	}

	hPrev := append([]float64(nil), h0.Data.Data()...)
	stackedData := make([]float64, T*H)

	for t := 0; t < T; t++ {
		xt := xData[t*I : (t+1)*I]
		im.xt[t] = append([]float64(nil), xt...)
		im.hPrev[t] = append([]float64(nil), hPrev...)

		// xGates [3H], hGates [3H]
		xGates := make([]float64, 3*H)
		hGates := make([]float64, 3*H)
		for m := 0; m < 3*H; m++ {
			sx := 0.0
			for n := 0; n < I; n++ {
				sx += xt[n] * wxData[n*3*H+m]
			}
			sh := 0.0
			for n := 0; n < H; n++ {
				sh += hPrev[n] * whData[n*3*H+m]
			}
			xGates[m] = sx
			hGates[m] = sh
		}
		im.hnInput[t] = append([]float64(nil), hGates[2*H:3*H]...)

		z := make([]float64, H)
		r := make([]float64, H)
		nVal := make([]float64, H)
		hNew := make([]float64, H)
		for j := 0; j < H; j++ {
			z[j] = sigmoidScalar(xGates[j] + hGates[j] + bData[j])
			r[j] = sigmoidScalar(xGates[H+j] + hGates[H+j] + bData[H+j])
			nInput := xGates[2*H+j] + bData[2*H+j] + r[j]*hGates[2*H+j]
			nVal[j] = math.Tanh(nInput)
			hNew[j] = (1-z[j])*hPrev[j] + z[j]*nVal[j]
		}
		im.z[t] = z
		im.r[t] = r
		im.n[t] = nVal

		copy(stackedData[t*H:(t+1)*H], hNew)
		hPrev = hNew
	}

	return autograd.NewResult(
		tensor.New(stackedData, []int{T, H}),
		&gruSequenceBackward{im: im, wxData: wxData, whData: whData},
		x, h0, WX, WH, B,
	)
}

type gruIntermediates struct {
	T, I, H int
	xt      [][]float64
	hPrev   [][]float64
	z, r, n [][]float64
	hnInput [][]float64 // hGates[2H:3H] per t (needed for dr backward)
}

type gruSequenceBackward struct {
	im     *gruIntermediates
	wxData []float64
	whData []float64
}

func (b *gruSequenceBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	im := b.im
	T, I, H := im.T, im.I, im.H
	gradFlat := grad.Data()

	dX := make([]float64, T*I)
	dWX := make([]float64, I*3*H)
	dWH := make([]float64, H*3*H)
	dB := make([]float64, 3*H)
	dhNext := make([]float64, H)

	for t := T - 1; t >= 0; t-- {
		z := im.z[t]
		r := im.r[t]
		nVal := im.n[t]
		hPrev := im.hPrev[t]
		xt := im.xt[t]
		hnIn := im.hnInput[t]

		dh := make([]float64, H)
		for j := 0; j < H; j++ {
			dh[j] = gradFlat[t*H+j] + dhNext[j]
		}

		// h_t = (1-z)*hPrev + z*n
		// dz = dh * (n - hPrev)
		// dn = dh * z
		// dhPrev_direct = dh * (1 - z)
		dLin := make([]float64, 3*H) // pre-activation gradients [dz_lin, dr_lin, dn_lin]
		dhPrevAcc := make([]float64, H)

		for j := 0; j < H; j++ {
			dz := dh[j] * (nVal[j] - hPrev[j])
			dn := dh[j] * z[j]
			dhPrevAcc[j] += dh[j] * (1 - z[j])

			// n = tanh(xn + b_n + r * hnIn)
			dnLin := dn * (1 - nVal[j]*nVal[j])
			dr := dnLin * hnIn[j]
			dHnIn := dnLin * r[j] // gradient into hGates[2H+j]

			dLin[0*H+j] = dz * z[j] * (1 - z[j]) // dz_lin
			dLin[1*H+j] = dr * r[j] * (1 - r[j]) // dr_lin
			dLin[2*H+j] = dnLin                  // dn pre-activation w.r.t. (xn + b_n)

			// dWH[:,2H+j] is split: hnIn contributes via dHnIn to hGates_n,
			// which equals hPrev @ WH_n. Apply this through WH below using a
			// separate path — accumulate dHnIn into a per-j buffer.
			// We handle it via dLinH (linear pre-activation for WH path).
			_ = dHnIn // see dLinH below
		}

		// For WH path, the "linear pre-activation" gradient is different for n gate
		// because hGates_n is multiplied by r before going into tanh:
		//   dHGates_n = dn_lin * r
		dLinH := make([]float64, 3*H)
		for j := 0; j < H; j++ {
			dLinH[0*H+j] = dLin[0*H+j]
			dLinH[1*H+j] = dLin[1*H+j]
			dLinH[2*H+j] = dLin[2*H+j] * r[j] // gated by r for WH side
		}

		// Bias: dB has same structure as xGates linear, since for n gate b_n
		// directly contributes to (xn + b_n + r*hn) — bias is NOT gated by r.
		for m := 0; m < 3*H; m++ {
			dB[m] += dLin[m]
		}

		// dWX, dxt — use dLin (x-side has no r-gating)
		for n := 0; n < I; n++ {
			s := 0.0
			for m := 0; m < 3*H; m++ {
				s += dLin[m] * b.wxData[n*3*H+m]
			}
			dX[t*I+n] = s
		}
		for n := 0; n < I; n++ {
			for m := 0; m < 3*H; m++ {
				dWX[n*3*H+m] += xt[n] * dLin[m]
			}
		}

		// dWH, dhPrev — use dLinH (n-gate gated by r)
		for n := 0; n < H; n++ {
			s := 0.0
			for m := 0; m < 3*H; m++ {
				s += dLinH[m] * b.whData[n*3*H+m]
			}
			dhPrevAcc[n] += s
		}
		for n := 0; n < H; n++ {
			for m := 0; m < 3*H; m++ {
				dWH[n*3*H+m] += hPrev[n] * dLinH[m]
			}
		}

		dhNext = dhPrevAcc
	}

	return []*tensor.Tensor{
		tensor.New(dX, []int{T, I}),
		tensor.New(dhNext, []int{H}),
		tensor.New(dWX, []int{I, 3 * H}),
		tensor.New(dWH, []int{H, 3 * H}),
		tensor.New(dB, []int{3 * H}),
	}
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


func sigmoidScalar(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// rowSlice extracts row t from a 2D Variable [T, H] as a 1D Variable [H],
// with proper autograd support: gradient on the slice flows back into the
// corresponding row of the source tensor.
type rowSliceBackward struct {
	t, T, H int
}

func (f *rowSliceBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	out := make([]float64, f.T*f.H)
	g := grad.Data()
	copy(out[f.t*f.H:(f.t+1)*f.H], g)
	return []*tensor.Tensor{tensor.New(out, []int{f.T, f.H})}
}

func rowSlice(x *autograd.Variable, t int) *autograd.Variable {
	T := x.Data.Shape()[0]
	H := x.Data.Shape()[1]
	row := make([]float64, H)
	copy(row, x.Data.Data()[t*H:(t+1)*H])
	return autograd.NewResult(
		tensor.New(row, []int{H}),
		&rowSliceBackward{t: t, T: T, H: H},
		x,
	)
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
