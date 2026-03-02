package nn

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// Conv2d implements a 2D convolution layer.
//
// Input:  [N, inC,  H,  W]
// Output: [N, outC, oH, oW]  where oH = (H + 2*P - kH) / S + 1
//
// Weight: [outC, inC, kH, kW]
// Bias:   [outC]
type Conv2d struct {
	InChannels  int
	OutChannels int
	KH, KW      int
	StrideH     int
	StrideW     int
	PadH        int
	PadW        int
	Weight      *autograd.Variable
	Bias        *autograd.Variable
	useBias     bool
}

// NewConv2d creates a Conv2d with He initialisation.
// kernelSize, stride, padding can be single ints (applied to both H and W).
func NewConv2d(inC, outC, kernelSize int, stride, padding int, bias bool) *Conv2d {
	fan := float64(inC * kernelSize * kernelSize)
	std := math.Sqrt(2.0 / fan)
	wSize := outC * inC * kernelSize * kernelSize
	wData := make([]float64, wSize)
	for i := range wData {
		wData[i] = rand.NormFloat64() * std
	}
	w := autograd.NewVar(tensor.New(wData, []int{outC, inC, kernelSize, kernelSize}), true)

	var b *autograd.Variable
	if bias {
		b = autograd.NewVar(tensor.Zeros(outC), true)
	}

	return &Conv2d{
		InChannels:  inC,
		OutChannels: outC,
		KH:          kernelSize,
		KW:          kernelSize,
		StrideH:     stride,
		StrideW:     stride,
		PadH:        padding,
		PadW:        padding,
		Weight:      w,
		Bias:        b,
		useBias:     bias,
	}
}

// Forward computes the 2D convolution using im2col + matmul.
// x shape: [N, inC, H, W]
func (c *Conv2d) Forward(x *autograd.Variable) *autograd.Variable {
	return conv2dForward(x, c.Weight, c.Bias, c.StrideH, c.StrideW, c.PadH, c.PadW)
}

func (c *Conv2d) Parameters() []*autograd.Variable {
	if c.useBias {
		return []*autograd.Variable{c.Weight, c.Bias}
	}
	return []*autograd.Variable{c.Weight}
}

func (c *Conv2d) ZeroGrad() {
	c.Weight.ZeroGrad()
	if c.useBias {
		c.Bias.ZeroGrad()
	}
}

// ---------------------------------------------------------------------------
// im2col convolution with autograd
// ---------------------------------------------------------------------------

type conv2dBackward struct {
	// saved for backward
	xCol    *tensor.Tensor // [N*oH*oW, inC*kH*kW]
	wFlat   *tensor.Tensor // [outC, inC*kH*kW]
	xShape  []int          // [N, inC, H, W]
	wShape  []int          // [outC, inC, kH, kW]
	strideH, strideW int
	padH, padW       int
	oH, oW           int
}

func (f *conv2dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	N, inC := f.xShape[0], f.xShape[1]
	H, W := f.xShape[2], f.xShape[3]
	kH, kW := f.wShape[2], f.wShape[3]
	outC := f.wShape[0]

	// grad shape: [N, outC, oH, oW]
	// Reshape to [N*oH*oW, outC]
	gradFlat := reshapeNHWCtoCols(grad, N, outC, f.oH, f.oW) // [N*oH*oW, outC]

	// dW = gradFlat^T @ xCol → [outC, inC*kH*kW]
	dW := tensor.MatMul(gradFlat.T(), f.xCol)
	dW = dW.Reshape(f.wShape...)

	// dXcol = gradFlat @ wFlat → [N*oH*oW, inC*kH*kW]
	dXcol := tensor.MatMul(gradFlat, f.wFlat)

	// col2im to get dX [N, inC, H, W]
	dX := col2im(dXcol, N, inC, H, W, kH, kW, f.strideH, f.strideW, f.padH, f.padW, f.oH, f.oW)

	return []*tensor.Tensor{dX, dW}
}

// conv2dForward performs im2col + matmul convolution and registers backward.
func conv2dForward(x, w, b *autograd.Variable, sH, sW, pH, pW int) *autograd.Variable {
	xd := x.Data
	wd := w.Data
	N, inC, H, W := xd.Shape()[0], xd.Shape()[1], xd.Shape()[2], xd.Shape()[3]
	outC, _, kH, kW := wd.Shape()[0], wd.Shape()[1], wd.Shape()[2], wd.Shape()[3]

	oH := (H+2*pH-kH)/sH + 1
	oW := (W+2*pW-kW)/sW + 1

	// im2col: [N*oH*oW, inC*kH*kW]
	xCol := im2col(xd, N, inC, H, W, kH, kW, sH, sW, pH, pW, oH, oW)

	// wFlat: [outC, inC*kH*kW]
	wFlat := wd.Reshape(outC, inC*kH*kW)

	// out = xCol @ wFlat^T → [N*oH*oW, outC]
	outFlat := tensor.MatMul(xCol, wFlat.T())

	// Reshape to [N, outC, oH, oW]
	outT := reshapeColsToNHWC(outFlat, N, outC, oH, oW)

	children := []*autograd.Variable{x, w}
	gradFn := &conv2dBackward{
		xCol: xCol, wFlat: wFlat,
		xShape: xd.Shape(), wShape: wd.Shape(),
		strideH: sH, strideW: sW, padH: pH, padW: pW,
		oH: oH, oW: oW,
	}
	result := autograd.NewResult(outT, gradFn, children...)

	// Add bias: [outC] broadcast over [N, outC, oH, oW]
	if b != nil {
		result = addConvBias(result, b, N, outC, oH, oW)
	}
	return result
}

// addConvBias adds bias [outC] to output [N, outC, oH, oW].
type addConvBiasBackward struct{ N, oH, oW int }

func (f *addConvBiasBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// dOutput = grad (pass through)
	// dBias = sum over N, oH, oW for each outC channel
	// grad shape: [N, outC, oH, oW]
	outC := grad.Shape()[1]
	dBias := tensor.Zeros(outC)
	for n := 0; n < f.N; n++ {
		for c := 0; c < outC; c++ {
			for h := 0; h < f.oH; h++ {
				for w := 0; w < f.oW; w++ {
					dBias.Set(dBias.At(c)+grad.At(n, c, h, w), c)
				}
			}
		}
	}
	return []*tensor.Tensor{grad, dBias}
}

func addConvBias(out, b *autograd.Variable, N, outC, oH, oW int) *autograd.Variable {
	// Broadcast bias to output shape
	outD := out.Data
	bd := b.Data
	result := tensor.Zeros(N, outC, oH, oW)
	for n := 0; n < N; n++ {
		for c := 0; c < outC; c++ {
			bv := bd.At(c)
			for h := 0; h < oH; h++ {
				for w := 0; w < oW; w++ {
					result.Set(outD.At(n, c, h, w)+bv, n, c, h, w)
				}
			}
		}
	}
	return autograd.NewResult(result, &addConvBiasBackward{N, oH, oW}, out, b)
}

// ---------------------------------------------------------------------------
// im2col / col2im helpers
// ---------------------------------------------------------------------------

// im2col converts input [N, inC, H, W] to column matrix [N*oH*oW, inC*kH*kW].
func im2col(x *tensor.Tensor, N, inC, H, W, kH, kW, sH, sW, pH, pW, oH, oW int) *tensor.Tensor {
	cols := tensor.Zeros(N*oH*oW, inC*kH*kW)
	row := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < oH; oh++ {
			for ow := 0; ow < oW; ow++ {
				col := 0
				for c := 0; c < inC; c++ {
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							ih := oh*sH + kh - pH
							iw := ow*sW + kw - pW
							var v float64
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								v = x.At(n, c, ih, iw)
							}
							cols.Set(v, row, col)
							col++
						}
					}
				}
				row++
			}
		}
	}
	return cols
}

// col2im converts gradient from column form [N*oH*oW, inC*kH*kW] back to [N, inC, H, W].
func col2im(dCol *tensor.Tensor, N, inC, H, W, kH, kW, sH, sW, pH, pW, oH, oW int) *tensor.Tensor {
	dx := tensor.Zeros(N, inC, H, W)
	row := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < oH; oh++ {
			for ow := 0; ow < oW; ow++ {
				col := 0
				for c := 0; c < inC; c++ {
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							ih := oh*sH + kh - pH
							iw := ow*sW + kw - pW
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								dx.Set(dx.At(n, c, ih, iw)+dCol.At(row, col), n, c, ih, iw)
							}
							col++
						}
					}
				}
				row++
			}
		}
	}
	return dx
}

// reshapeColsToNHWC converts [N*oH*oW, outC] → [N, outC, oH, oW].
func reshapeColsToNHWC(m *tensor.Tensor, N, outC, oH, oW int) *tensor.Tensor {
	out := tensor.Zeros(N, outC, oH, oW)
	row := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < oH; oh++ {
			for ow := 0; ow < oW; ow++ {
				for c := 0; c < outC; c++ {
					out.Set(m.At(row, c), n, c, oh, ow)
				}
				row++
			}
		}
	}
	return out
}

// reshapeNHWCtoCols converts [N, outC, oH, oW] → [N*oH*oW, outC].
func reshapeNHWCtoCols(m *tensor.Tensor, N, outC, oH, oW int) *tensor.Tensor {
	out := tensor.Zeros(N*oH*oW, outC)
	row := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < oH; oh++ {
			for ow := 0; ow < oW; ow++ {
				for c := 0; c < outC; c++ {
					out.Set(m.At(n, c, oh, ow), row, c)
				}
				row++
			}
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// MaxPool2d
// ---------------------------------------------------------------------------

// MaxPool2d performs 2D max pooling.
// Input:  [N, C, H,  W]
// Output: [N, C, oH, oW]  where oH = (H - kH) / S + 1
type MaxPool2d struct {
	KH, KW      int
	StrideH     int
	StrideW     int
}

// NewMaxPool2d creates a MaxPool2d. stride defaults to kernelSize if 0.
func NewMaxPool2d(kernelSize, stride int) *MaxPool2d {
	if stride <= 0 {
		stride = kernelSize
	}
	return &MaxPool2d{KH: kernelSize, KW: kernelSize, StrideH: stride, StrideW: stride}
}

func (p *MaxPool2d) Parameters() []*autograd.Variable { return nil }
func (p *MaxPool2d) ZeroGrad()                         {}

func (p *MaxPool2d) Forward(x *autograd.Variable) *autograd.Variable {
	return maxPool2dForward(x, p.KH, p.KW, p.StrideH, p.StrideW)
}

type maxPool2dBackward struct {
	mask   *tensor.Tensor // 1 at max position, 0 elsewhere; shape [N,C,H,W]
	xShape []int
	oH, oW int
	kH, kW int
	sH, sW int
}

func (f *maxPool2dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	N, C := f.xShape[0], f.xShape[1]
	H, W := f.xShape[2], f.xShape[3]
	dx := tensor.Zeros(N, C, H, W)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < f.oH; oh++ {
				for ow := 0; ow < f.oW; ow++ {
					g := grad.At(n, c, oh, ow)
					for kh := 0; kh < f.kH; kh++ {
						for kw := 0; kw < f.kW; kw++ {
							ih := oh*f.sH + kh
							iw := ow*f.sW + kw
							if ih < H && iw < W && f.mask.At(n, c, ih, iw) != 0 {
								dx.Set(dx.At(n, c, ih, iw)+g, n, c, ih, iw)
							}
						}
					}
				}
			}
		}
	}
	return []*tensor.Tensor{dx}
}

func maxPool2dForward(x *autograd.Variable, kH, kW, sH, sW int) *autograd.Variable {
	xd := x.Data
	N, C, H, W := xd.Shape()[0], xd.Shape()[1], xd.Shape()[2], xd.Shape()[3]
	oH := (H-kH)/sH + 1
	oW := (W-kW)/sW + 1

	out := tensor.Zeros(N, C, oH, oW)
	mask := tensor.Zeros(N, C, H, W)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < oH; oh++ {
				for ow := 0; ow < oW; ow++ {
					maxVal := math.Inf(-1)
					maxIh, maxIw := 0, 0
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							ih := oh*sH + kh
							iw := ow*sW + kw
							v := xd.At(n, c, ih, iw)
							if v > maxVal {
								maxVal = v
								maxIh, maxIw = ih, iw
							}
						}
					}
					out.Set(maxVal, n, c, oh, ow)
					mask.Set(1.0, n, c, maxIh, maxIw)
				}
			}
		}
	}

	return autograd.NewResult(out, &maxPool2dBackward{
		mask: mask, xShape: xd.Shape(),
		oH: oH, oW: oW, kH: kH, kW: kW, sH: sH, sW: sW,
	}, x)
}

// Flatten2d flattens [N, C, H, W] → [N, C*H*W] for the FC layers after conv.
func Flatten2d(x *autograd.Variable) *autograd.Variable {
	shape := x.Data.Shape()
	if len(shape) != 4 {
		panic(fmt.Sprintf("nn.Flatten2d: expected 4D input, got shape %v", shape))
	}
	N := shape[0]
	flat := shape[1] * shape[2] * shape[3]
	return autograd.NewResult(
		x.Data.Reshape(N, flat),
		&flatten2dBackward{origShape: shape},
		x,
	)
}

type flatten2dBackward struct{ origShape []int }

func (f *flatten2dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	return []*tensor.Tensor{grad.Reshape(f.origShape...)}
}
