package nn

import (
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ---------------------------------------------------------------------------
// BatchNorm2d
// ---------------------------------------------------------------------------

// BatchNorm2d normalises over the channel dimension for 4D input [N, C, H, W].
//
// During training: normalises using batch statistics, updates running mean/var.
// During eval:     uses running mean/var (no batch dependency).
//
// Learnable: gamma (scale) and beta (shift), both shape [C].
type BatchNorm2d struct {
	NumFeatures int
	Eps         float64
	Momentum    float64 // for running stats update

	Gamma       *autograd.Variable // scale, init 1
	Beta        *autograd.Variable // shift, init 0

	RunningMean []float64
	RunningVar  []float64
	Training    bool
}

// NewBatchNorm2d creates a BatchNorm2d for C channels.
func NewBatchNorm2d(numFeatures int) *BatchNorm2d {
	ones := make([]float64, numFeatures)
	zeros := make([]float64, numFeatures)
	for i := range ones {
		ones[i] = 1.0
	}
	return &BatchNorm2d{
		NumFeatures: numFeatures,
		Eps:         1e-5,
		Momentum:    0.1,
		Gamma:       autograd.NewVar(tensor.New(ones, []int{numFeatures}), true),
		Beta:        autograd.NewVar(tensor.New(zeros, []int{numFeatures}), true),
		RunningMean: make([]float64, numFeatures),
		RunningVar:  makeFilled(numFeatures, 1.0),
		Training:    true,
	}
}

func makeFilled(n int, v float64) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = v
	}
	return s
}

func (b *BatchNorm2d) Train() { b.Training = true }
func (b *BatchNorm2d) Eval()  { b.Training = false }

func (b *BatchNorm2d) Parameters() []*autograd.Variable {
	return []*autograd.Variable{b.Gamma, b.Beta}
}
func (b *BatchNorm2d) ZeroGrad() {
	b.Gamma.ZeroGrad()
	b.Beta.ZeroGrad()
}

// Forward normalises x [N, C, H, W].
func (b *BatchNorm2d) Forward(x *autograd.Variable) *autograd.Variable {
	return batchNorm2dForward(x, b.Gamma, b.Beta, b, b.Training)
}

type batchNorm2dBackward struct {
	xNorm    *tensor.Tensor   // [N, C, H, W] normalised input
	gamma    *tensor.Tensor   // [C]
	mean     []float64        // [C]
	variance []float64        // [C]
	N, C, H, W int
	eps      float64
}

func (f *batchNorm2dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	N, C, H, W := f.N, f.C, f.H, f.W
	M := float64(N * H * W)

	dGamma := tensor.Zeros(C)
	dBeta := tensor.Zeros(C)
	dX := tensor.Zeros(N, C, H, W)

	for c := 0; c < C; c++ {
		// Sum grad and grad*xNorm over N,H,W for this channel
		sumDy := 0.0
		sumDyXn := 0.0
		gv := f.gamma.At(c)
		std := math.Sqrt(f.variance[c] + f.eps)

		for n := 0; n < N; n++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					dy := grad.At(n, c, h, w)
					xn := f.xNorm.At(n, c, h, w)
					sumDy += dy
					sumDyXn += dy * xn
					dGamma.Set(dGamma.At(c)+dy*xn, c)
					dBeta.Set(dBeta.At(c)+dy, c)
				}
			}
		}

		// dX per element: (1/std) * gamma * (dy - sumDy/M - xn*sumDyXn/M)
		for n := 0; n < N; n++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					dy := grad.At(n, c, h, w)
					xn := f.xNorm.At(n, c, h, w)
					dx := (gv / std) * (dy - sumDy/M - xn*sumDyXn/M)
					dX.Set(dx, n, c, h, w)
				}
			}
		}
	}

	// Return gradients for [x, gamma, beta]
	return []*tensor.Tensor{dX, dGamma, dBeta}
}

func batchNorm2dForward(x, gamma, beta *autograd.Variable, bn *BatchNorm2d, training bool) *autograd.Variable {
	xd := x.Data
	shape := xd.Shape()
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	M := float64(N * H * W)

	mean := make([]float64, C)
	variance := make([]float64, C)

	if training {
		// Compute batch mean and variance per channel
		for c := 0; c < C; c++ {
			sum := 0.0
			for n := 0; n < N; n++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						sum += xd.At(n, c, h, w)
					}
				}
			}
			mean[c] = sum / M

			varSum := 0.0
			for n := 0; n < N; n++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						d := xd.At(n, c, h, w) - mean[c]
						varSum += d * d
					}
				}
			}
			variance[c] = varSum / M

			// Update running stats
			bn.RunningMean[c] = (1-bn.Momentum)*bn.RunningMean[c] + bn.Momentum*mean[c]
			bn.RunningVar[c] = (1-bn.Momentum)*bn.RunningVar[c] + bn.Momentum*variance[c]
		}
	} else {
		// Use running stats in eval mode
		copy(mean, bn.RunningMean)
		copy(variance, bn.RunningVar)
	}

	// Normalise
	xNorm := tensor.Zeros(N, C, H, W)
	out := tensor.Zeros(N, C, H, W)
	gd := gamma.Data.Data()
	bd := beta.Data.Data()

	for c := 0; c < C; c++ {
		std := math.Sqrt(variance[c] + bn.Eps)
		for n := 0; n < N; n++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					xn := (xd.At(n, c, h, w) - mean[c]) / std
					xNorm.Set(xn, n, c, h, w)
					out.Set(gd[c]*xn+bd[c], n, c, h, w)
				}
			}
		}
	}

	return autograd.NewResult(out, &batchNorm2dBackward{
		xNorm: xNorm, gamma: gamma.Data,
		mean: mean, variance: variance,
		N: N, C: C, H: H, W: W, eps: bn.Eps,
	}, x, gamma, beta)
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

// LayerNorm normalises over the last D dimensions.
// normalizedShape = the shape of the dimensions to normalise over.
//
// For transformers typically: LayerNorm([dModel]) applied to [..., dModel].
type LayerNorm struct {
	NormalizedShape []int
	Eps             float64
	Gamma           *autograd.Variable
	Beta            *autograd.Variable
}

func NewLayerNorm(normalizedShape []int) *LayerNorm {
	size := 1
	for _, d := range normalizedShape {
		size *= d
	}
	return &LayerNorm{
		NormalizedShape: normalizedShape,
		Eps:             1e-5,
		Gamma:           autograd.NewVar(tensor.Ones(normalizedShape...), true),
		Beta:            autograd.NewVar(tensor.Zeros(normalizedShape...), true),
	}
}

func (l *LayerNorm) Parameters() []*autograd.Variable {
	return []*autograd.Variable{l.Gamma, l.Beta}
}
func (l *LayerNorm) ZeroGrad() { l.Gamma.ZeroGrad(); l.Beta.ZeroGrad() }

func (l *LayerNorm) Forward(x *autograd.Variable) *autograd.Variable {
	return layerNormForward(x, l.Gamma, l.Beta, l.Eps)
}

type layerNormBackward struct {
	xNorm    *tensor.Tensor
	gamma    *tensor.Tensor
	normSize int
	eps      float64
	outShape []int
	stds     []float64 // per-group std, captured in forward
}

func (f *layerNormBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	M := float64(f.normSize)
	total := grad.Size()
	groups := total / f.normSize

	gradFlat := grad.Data()
	xnFlat := f.xNorm.Data()
	gammaFlat := f.gamma.Data()

	dGammaFlat := make([]float64, f.normSize)
	dBetaFlat := make([]float64, f.normSize)
	dXFlat := make([]float64, total)

	for g := 0; g < groups; g++ {
		base := g * f.normSize
		std := f.stds[g]

		sumDy := 0.0
		sumDyXn := 0.0
		for j := 0; j < f.normSize; j++ {
			dy := gradFlat[base+j]
			xn := xnFlat[base+j]
			sumDy += dy * gammaFlat[j]
			sumDyXn += dy * gammaFlat[j] * xn
			dGammaFlat[j] += dy * xn
			dBetaFlat[j] += dy
		}
		for j := 0; j < f.normSize; j++ {
			dy := gradFlat[base+j]
			xn := xnFlat[base+j]
			// dx = (1/std) * (gamma*dy - sumDy/M - xn*sumDyXn/M)
			dXFlat[base+j] = (gammaFlat[j]*dy - sumDy/M - xn*sumDyXn/M) / std
		}
	}

	dX := tensor.New(dXFlat, f.outShape)
	dGamma := tensor.New(dGammaFlat, f.gamma.Shape())
	dBeta := tensor.New(dBetaFlat, f.gamma.Shape())
	return []*tensor.Tensor{dX, dGamma, dBeta}
}

func layerNormForward(x, gamma, beta *autograd.Variable, eps float64) *autograd.Variable {
	xd := x.Data
	shape := xd.Shape()
	normSize := gamma.Data.Size()
	total := xd.Size()
	groups := total / normSize

	xFlat := xd.Data()
	gFlat := gamma.Data.Data()
	bFlat := beta.Data.Data()

	outFlat := make([]float64, total)
	xnFlat := make([]float64, total)
	stds := make([]float64, groups)

	for g := 0; g < groups; g++ {
		base := g * normSize
		// Mean
		sum := 0.0
		for j := 0; j < normSize; j++ {
			sum += xFlat[base+j]
		}
		mean := sum / float64(normSize)
		// Var
		varSum := 0.0
		for j := 0; j < normSize; j++ {
			d := xFlat[base+j] - mean
			varSum += d * d
		}
		std := math.Sqrt(varSum/float64(normSize) + eps)
		stds[g] = std
		// Normalise
		for j := 0; j < normSize; j++ {
			xn := (xFlat[base+j] - mean) / std
			xnFlat[base+j] = xn
			outFlat[base+j] = gFlat[j]*xn + bFlat[j]
		}
	}

	out := tensor.New(outFlat, shape)
	xNorm := tensor.New(xnFlat, shape)

	return autograd.NewResult(out, &layerNormBackward{
		xNorm: xNorm, gamma: gamma.Data,
		normSize: normSize, eps: eps, outShape: shape,
		stds: stds,
	}, x, gamma, beta)
}

// ---------------------------------------------------------------------------
// BatchNorm1d
// ---------------------------------------------------------------------------

// BatchNorm1d normalises over the feature dimension for 2D [N, C] or
// 3D [N, C, L] input (sequences).
//
// During training: batch statistics; Eval: running stats.
// Learnable: gamma [C], beta [C].
type BatchNorm1d struct {
	NumFeatures int
	Eps         float64
	Momentum    float64

	Gamma       *autograd.Variable
	Beta        *autograd.Variable

	RunningMean []float64
	RunningVar  []float64
	Training    bool
}

func NewBatchNorm1d(numFeatures int) *BatchNorm1d {
	ones := makeFilled(numFeatures, 1.0)
	zeros := make([]float64, numFeatures)
	return &BatchNorm1d{
		NumFeatures: numFeatures,
		Eps:         1e-5,
		Momentum:    0.1,
		Gamma:       autograd.NewVar(tensor.New(ones, []int{numFeatures}), true),
		Beta:        autograd.NewVar(tensor.New(zeros, []int{numFeatures}), true),
		RunningMean: make([]float64, numFeatures),
		RunningVar:  makeFilled(numFeatures, 1.0),
		Training:    true,
	}
}

func (b *BatchNorm1d) Train() { b.Training = true }
func (b *BatchNorm1d) Eval()  { b.Training = false }

func (b *BatchNorm1d) Parameters() []*autograd.Variable {
	return []*autograd.Variable{b.Gamma, b.Beta}
}
func (b *BatchNorm1d) ZeroGrad() { b.Gamma.ZeroGrad(); b.Beta.ZeroGrad() }

// Forward accepts [N, C] or [N, C, L].
func (b *BatchNorm1d) Forward(x *autograd.Variable) *autograd.Variable {
	shape := x.Data.Shape()
	C := b.NumFeatures

	// Determine M = number of elements averaged per channel
	// [N, C] → M = N; [N, C, L] → M = N*L
	var N, L int
	switch len(shape) {
	case 2:
		N, L = shape[0], 1
	case 3:
		N, L = shape[0], shape[2]
	default:
		panic("BatchNorm1d: expected 2D [N,C] or 3D [N,C,L] input")
	}
	M := float64(N * L)

	xFlat := x.Data.Data()
	mean := make([]float64, C)
	variance := make([]float64, C)

	if b.Training {
		for c := 0; c < C; c++ {
			sum := 0.0
			for n := 0; n < N; n++ {
				for l := 0; l < L; l++ {
					sum += xFlat[n*C*L+c*L+l]
				}
			}
			mean[c] = sum / M

			varSum := 0.0
			for n := 0; n < N; n++ {
				for l := 0; l < L; l++ {
					d := xFlat[n*C*L+c*L+l] - mean[c]
					varSum += d * d
				}
			}
			variance[c] = varSum / M

			b.RunningMean[c] = (1-b.Momentum)*b.RunningMean[c] + b.Momentum*mean[c]
			b.RunningVar[c] = (1-b.Momentum)*b.RunningVar[c] + b.Momentum*variance[c]
		}
	} else {
		copy(mean, b.RunningMean)
		copy(variance, b.RunningVar)
	}

	outFlat := make([]float64, len(xFlat))
	gd := b.Gamma.Data.Data()
	bd := b.Beta.Data.Data()

	for c := 0; c < C; c++ {
		std := math.Sqrt(variance[c] + b.Eps)
		for n := 0; n < N; n++ {
			for l := 0; l < L; l++ {
				idx := n*C*L + c*L + l
				xn := (xFlat[idx] - mean[c]) / std
				outFlat[idx] = gd[c]*xn + bd[c]
			}
		}
	}

	out := tensor.New(outFlat, shape)

	// Store xnorm and stds for backward.
	xnFlat := make([]float64, len(xFlat))
	stds := make([]float64, C)
	for c := 0; c < C; c++ {
		std := math.Sqrt(variance[c] + b.Eps)
		stds[c] = std
		for n := 0; n < N; n++ {
			for l := 0; l < L; l++ {
				idx := n*C*L + c*L + l
				xnFlat[idx] = (xFlat[idx] - mean[c]) / std
			}
		}
	}
	xNorm := tensor.New(xnFlat, shape)

	return autograd.NewResult(out, &batchNorm1dBackward{
		xNorm:    xNorm,
		gamma:    b.Gamma.Data,
		stds:     stds,
		N:        N,
		C:        C,
		L:        L,
		training: b.Training,
		outShape: shape,
	}, x, b.Gamma, b.Beta)
}

type batchNorm1dBackward struct {
	xNorm    *tensor.Tensor
	gamma    *tensor.Tensor
	stds     []float64 // per channel
	N, C, L  int
	training bool
	outShape []int
}

func (f *batchNorm1dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gradFlat := grad.Data()
	xnFlat := f.xNorm.Data()
	gammaFlat := f.gamma.Data()

	dGammaFlat := make([]float64, f.C)
	dBetaFlat := make([]float64, f.C)
	dXFlat := make([]float64, len(gradFlat))

	M := float64(f.N * f.L)

	for c := 0; c < f.C; c++ {
		std := f.stds[c]
		sumDy := 0.0
		sumDyXn := 0.0
		for n := 0; n < f.N; n++ {
			for l := 0; l < f.L; l++ {
				idx := n*f.C*f.L + c*f.L + l
				dy := gradFlat[idx]
				xn := xnFlat[idx]
				sumDy += dy
				sumDyXn += dy * xn
				dGammaFlat[c] += dy * xn
				dBetaFlat[c] += dy
			}
		}
		for n := 0; n < f.N; n++ {
			for l := 0; l < f.L; l++ {
				idx := n*f.C*f.L + c*f.L + l
				dy := gradFlat[idx]
				xn := xnFlat[idx]
				if f.training {
					// dx = (gamma/std) * (dy - sumDy/M - xn*sumDyXn/M)
					dXFlat[idx] = (gammaFlat[c] / std) * (dy - sumDy/M - xn*sumDyXn/M)
				} else {
					// eval: std is a constant
					dXFlat[idx] = (gammaFlat[c] / std) * dy
				}
			}
		}
	}

	return []*tensor.Tensor{
		tensor.New(dXFlat, f.outShape),
		tensor.New(dGammaFlat, f.gamma.Shape()),
		tensor.New(dBetaFlat, f.gamma.Shape()),
	}
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

// GroupNorm divides C channels into numGroups groups and normalises
// within each group independently.
//
// Input: [N, C, H, W] (4D) or [N, C, L] (3D) or [N, C] (2D).
// Learnable: gamma [C], beta [C].
type GroupNorm struct {
	NumGroups   int
	NumChannels int
	Eps         float64
	Gamma       *autograd.Variable // [C]
	Beta        *autograd.Variable // [C]
}

func NewGroupNorm(numGroups, numChannels int) *GroupNorm {
	if numChannels%numGroups != 0 {
		panic("GroupNorm: numChannels must be divisible by numGroups")
	}
	return &GroupNorm{
		NumGroups:   numGroups,
		NumChannels: numChannels,
		Eps:         1e-5,
		Gamma:       autograd.NewVar(tensor.Ones(numChannels), true),
		Beta:        autograd.NewVar(tensor.Zeros(numChannels), true),
	}
}

func (g *GroupNorm) Parameters() []*autograd.Variable {
	return []*autograd.Variable{g.Gamma, g.Beta}
}
func (g *GroupNorm) ZeroGrad() { g.Gamma.ZeroGrad(); g.Beta.ZeroGrad() }

// Forward normalises x [N, C, ...] within each channel group.
func (gn *GroupNorm) Forward(x *autograd.Variable) *autograd.Variable {
	shape := x.Data.Shape()
	N, C := shape[0], shape[1]
	G := gn.NumGroups
	chPerGroup := C / G

	// Total spatial elements per channel
	spatial := 1
	for _, d := range shape[2:] {
		spatial *= d
	}
	elemPerGroup := chPerGroup * spatial // elements per group per sample

	xFlat := x.Data.Data()
	outFlat := make([]float64, len(xFlat))
	xnFlat := make([]float64, len(xFlat))
	gd := gn.Gamma.Data.Data()
	bd := gn.Beta.Data.Data()

	// Stds keyed by (n, group) for backward.
	stds := make([]float64, N*G)

	for n := 0; n < N; n++ {
		for gr := 0; gr < G; gr++ {
			cStart := gr * chPerGroup
			cEnd := cStart + chPerGroup

			sum := 0.0
			for c := cStart; c < cEnd; c++ {
				for s := 0; s < spatial; s++ {
					sum += xFlat[n*C*spatial+c*spatial+s]
				}
			}
			mean := sum / float64(elemPerGroup)

			varSum := 0.0
			for c := cStart; c < cEnd; c++ {
				for s := 0; s < spatial; s++ {
					d := xFlat[n*C*spatial+c*spatial+s] - mean
					varSum += d * d
				}
			}
			std := math.Sqrt(varSum/float64(elemPerGroup) + gn.Eps)
			stds[n*G+gr] = std

			for c := cStart; c < cEnd; c++ {
				for s := 0; s < spatial; s++ {
					idx := n*C*spatial + c*spatial + s
					xn := (xFlat[idx] - mean) / std
					xnFlat[idx] = xn
					outFlat[idx] = gd[c]*xn + bd[c]
				}
			}
		}
	}

	out := tensor.New(outFlat, shape)
	xNorm := tensor.New(xnFlat, shape)

	return autograd.NewResult(out, &groupNormBackward{
		xNorm:        xNorm,
		gamma:        gn.Gamma.Data,
		stds:         stds,
		N:            N,
		C:            C,
		G:            G,
		chPerGroup:   chPerGroup,
		spatial:      spatial,
		elemPerGroup: elemPerGroup,
		outShape:     shape,
	}, x, gn.Gamma, gn.Beta)
}

type groupNormBackward struct {
	xNorm                                       *tensor.Tensor
	gamma                                       *tensor.Tensor
	stds                                        []float64
	N, C, G, chPerGroup, spatial, elemPerGroup  int
	outShape                                    []int
}

func (f *groupNormBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gradFlat := grad.Data()
	xnFlat := f.xNorm.Data()
	gammaFlat := f.gamma.Data()

	dGammaFlat := make([]float64, f.C)
	dBetaFlat := make([]float64, f.C)
	dXFlat := make([]float64, len(gradFlat))

	M := float64(f.elemPerGroup)

	for n := 0; n < f.N; n++ {
		for gr := 0; gr < f.G; gr++ {
			cStart := gr * f.chPerGroup
			cEnd := cStart + f.chPerGroup
			std := f.stds[n*f.G+gr]

			sumDyG := 0.0   // sum of (dy * gamma) over group
			sumDyGXn := 0.0 // sum of (dy * gamma * xn) over group
			for c := cStart; c < cEnd; c++ {
				for s := 0; s < f.spatial; s++ {
					idx := n*f.C*f.spatial + c*f.spatial + s
					dy := gradFlat[idx]
					xn := xnFlat[idx]
					sumDyG += dy * gammaFlat[c]
					sumDyGXn += dy * gammaFlat[c] * xn
					dGammaFlat[c] += dy * xn
					dBetaFlat[c] += dy
				}
			}
			for c := cStart; c < cEnd; c++ {
				for s := 0; s < f.spatial; s++ {
					idx := n*f.C*f.spatial + c*f.spatial + s
					dy := gradFlat[idx]
					xn := xnFlat[idx]
					dXFlat[idx] = (gammaFlat[c]*dy - sumDyG/M - xn*sumDyGXn/M) / std
				}
			}
		}
	}

	return []*tensor.Tensor{
		tensor.New(dXFlat, f.outShape),
		tensor.New(dGammaFlat, f.gamma.Shape()),
		tensor.New(dBetaFlat, f.gamma.Shape()),
	}
}
