package nn

import (
	"fmt"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// DataParallel wraps a Module for multi-GPU data-parallel training.
//
// # Honest semantics
//
// With a single device ID (the common case on a one-GPU machine):
//
//	dp := nn.NewDataParallel(model, []int{0})
//	out := dp.Forward(x) // identical to model.Forward(x)
//
// DataParallel is a transparent pass-through; there is no overhead.
//
// With multiple device IDs (requires ≥2 physical CUDA GPUs):
//
//	dp := nn.NewDataParallel(model, []int{0, 1})
//
//  1. Input batch is split along dim-0 across devices.
//  2. Each shard is forwarded on its device concurrently.
//  3. Outputs are gathered (concatenated) on device DeviceIDs[0].
//  4. Gradients from each replica are summed back to the primary replica.
//
// NOTE: The current implementation runs shards sequentially on the CPU/primary
// GPU because GoTorch does not yet have per-device CUDA stream management.
// The API is stable and semantically correct; true concurrent dispatch will be
// added in a later version without breaking the interface.
type DataParallel struct {
	Module    Module
	DeviceIDs []int
}

// NewDataParallel wraps m for parallel execution across deviceIDs.
// deviceIDs must contain at least one element; the primary device is deviceIDs[0].
func NewDataParallel(m Module, deviceIDs []int) *DataParallel {
	if len(deviceIDs) == 0 {
		panic("nn.DataParallel: deviceIDs must not be empty")
	}
	return &DataParallel{Module: m, DeviceIDs: deviceIDs}
}

// Forward runs the model. Single-GPU: identity delegation. Multi-GPU: split → forward × N → gather.
func (dp *DataParallel) Forward(x *autograd.Variable) *autograd.Variable {
	if len(dp.DeviceIDs) == 1 {
		return dp.Module.Forward(x)
	}
	return dp.parallelForward(x)
}

// parallelForward implements split → replicated forward → gather.
func (dp *DataParallel) parallelForward(x *autograd.Variable) *autograd.Variable {
	n := len(dp.DeviceIDs)
	batchSize := x.Data.Shape()[0]

	if batchSize < n {
		// Not enough rows to shard — fall back to primary device.
		return dp.Module.Forward(x)
	}

	// ── Split ────────────────────────────────────────────────────────────────
	shards := splitBatch(x, n)

	// ── Forward on each shard (sequential; see doc comment) ──────────────────
	results := make([]*autograd.Variable, n)
	for i, shard := range shards {
		results[i] = dp.Module.Forward(shard)
	}

	// ── Gather ───────────────────────────────────────────────────────────────
	return gatherBatch(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameters / ZeroGrad — delegate to the wrapped module
// ─────────────────────────────────────────────────────────────────────────────

func (dp *DataParallel) Parameters() []*autograd.Variable { return dp.Module.Parameters() }
func (dp *DataParallel) ZeroGrad()                        { dp.Module.ZeroGrad() }

func (dp *DataParallel) String() string {
	return fmt.Sprintf("DataParallel(devices=%v, module=%T)", dp.DeviceIDs, dp.Module)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

// splitBatch divides x along dim-0 into n roughly equal shards.
func splitBatch(x *autograd.Variable, n int) []*autograd.Variable {
	shape := x.Data.Shape()
	batchSize := shape[0]
	cols := 1
	for i := 1; i < len(shape); i++ {
		cols *= shape[i]
	}

	shards := make([]*autograd.Variable, n)
	src := x.Data.Data()
	base := batchSize / n

	offset := 0
	for i := 0; i < n; i++ {
		rows := base
		if i < batchSize%n {
			rows++ // distribute remainder
		}
		d := make([]float64, rows*cols)
		copy(d, src[offset*cols:(offset+rows)*cols])
		offset += rows

		newShape := append([]int{rows}, shape[1:]...)
		shards[i] = autograd.NewVar(tensor.New(d, newShape), x.RequiresGrad)
	}
	return shards
}

// gatherBatch concatenates variables along dim-0 and builds a differentiable node.
func gatherBatch(parts []*autograd.Variable) *autograd.Variable {
	if len(parts) == 1 {
		return parts[0]
	}
	shape0 := parts[0].Data.Shape()
	cols := 1
	for i := 1; i < len(shape0); i++ {
		cols *= shape0[i]
	}

	totalRows := 0
	for _, p := range parts {
		totalRows += p.Data.Shape()[0]
	}

	out := make([]float64, totalRows*cols)
	offset := 0
	for _, p := range parts {
		rows := p.Data.Shape()[0]
		copy(out[offset:], p.Data.Data())
		offset += rows * cols
	}

	newShape := append([]int{totalRows}, shape0[1:]...)
	outT := tensor.New(out, newShape)

	inputs := make([]*autograd.Variable, len(parts))
	copy(inputs, parts)

	return autograd.NewResult(outT, &gatherBwd{parts: inputs, cols: cols}, inputs...)
}

// gatherBwd scatters the upstream gradient back to each shard.
type gatherBwd struct {
	parts []*autograd.Variable
	cols  int
}

func (g *gatherBwd) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	gradData := grad.Data()
	out := make([]*tensor.Tensor, len(g.parts))
	offset := 0
	for i, p := range g.parts {
		rows := p.Data.Shape()[0]
		n := rows * g.cols
		slice := make([]float64, n)
		copy(slice, gradData[offset:offset+n])
		offset += n
		out[i] = tensor.New(slice, p.Data.Shape())
	}
	return out
}
