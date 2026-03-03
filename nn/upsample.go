package nn

import (
	"fmt"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// Upsample upsamples spatial dimensions of a 4D tensor [N, C, H, W].
// Supported modes: "nearest"
type Upsample struct {
	ScaleFactor int
	Mode        string
}

// NewUpsample creates an Upsample layer.
// scaleFactor must be a positive integer.
// mode: "nearest" (default, only mode currently supported).
func NewUpsample(scaleFactor int, mode string) *Upsample {
	if scaleFactor <= 0 {
		panic("nn.Upsample: scaleFactor must be > 0")
	}
	if mode != "nearest" && mode != "bilinear" {
		panic(fmt.Sprintf("nn.Upsample: unsupported mode %q (supported: nearest)", mode))
	}
	return &Upsample{ScaleFactor: scaleFactor, Mode: mode}
}

func (u *Upsample) Parameters() []*autograd.Variable { return nil }
func (u *Upsample) ZeroGrad()                         {}

// Forward upsamples x: [N, C, H, W] → [N, C, H*scale, W*scale].
func (u *Upsample) Forward(x *autograd.Variable) *autograd.Variable {
	return upsampleForward(x, u.ScaleFactor)
}

// ── Nearest-neighbour upsample ────────────────────────────────────────────────

type upsampleBackward struct {
	inShape []int
	scale   int
}

func (f *upsampleBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// Downsample gradient by averaging over scale×scale blocks.
	N, C, H, W := f.inShape[0], f.inShape[1], f.inShape[2], f.inShape[3]
	s := f.scale
	dX := tensor.Zeros(N, C, H, W)
	dXd := dX.Data()
	inShape := f.inShape
	outShape := []int{N, C, H * s, W * s}

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					sum := 0.0
					for dh := 0; dh < s; dh++ {
						for dw := 0; dw < s; dw++ {
							oh := h*s + dh
							ow := w*s + dw
							sum += grad.At(n, c, oh, ow)
							_ = outShape
						}
					}
					dXd[flatIdx4(n, c, h, w, inShape)] = sum
				}
			}
		}
	}
	return []*tensor.Tensor{dX}
}

func upsampleForward(x *autograd.Variable, scale int) *autograd.Variable {
	xShape := x.Data.Shape() // [N, C, H, W]
	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	oH := H * scale
	oW := W * scale
	outShape := []int{N, C, oH, oW}
	outData := make([]float64, N*C*oH*oW)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					v := x.Data.At(n, c, h, w)
					for dh := 0; dh < scale; dh++ {
						for dw := 0; dw < scale; dw++ {
							oh := h*scale + dh
							ow := w*scale + dw
							outData[flatIdx4(n, c, oh, ow, outShape)] = v
						}
					}
				}
			}
		}
	}

	outT := tensor.New(outData, outShape)
	return autograd.NewResult(outT, &upsampleBackward{xShape, scale}, x)
}
