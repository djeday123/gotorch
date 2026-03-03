// Package export provides model serialization utilities for GoTorch models.
//
// Currently supports ONNX export for Sequential models containing Linear,
// ReLU, Sigmoid, Tanh, GELU, and Dropout layers.
//
// ONNX is written as valid protobuf without any external dependencies —
// a minimal subset of the wire format is implemented inline.
package export

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/djeday123/gotorch/nn"
)

// ExportONNX exports a *nn.Sequential model to an ONNX file at path.
//
// inputShape is the shape of one input sample (e.g. []int{1, 784} for a
// batch-of-1 with 784 features). The batch dimension is treated as dynamic.
//
// Supported layer types: *nn.Linear, *nn.ReLU, *nn.Sigmoid, *nn.Tanh,
// *nn.GELU, *nn.Dropout.
//
// Example:
//
//	model := nn.NewSequential(
//	    nn.NewLinear(784, 256, true),
//	    nn.NewReLU(),
//	    nn.NewLinear(256, 10, true),
//	)
//	err := export.ExportONNX(model, []int{1, 784}, "model.onnx")
func ExportONNX(model *nn.Sequential, inputShape []int, path string) error {
	g := &onnxGraph{
		nodes:        nil,
		initializers: nil,
		inputName:    "input_0",
		outputName:   "",
	}

	cur := "input_0"
	nodeIdx := 0

	for i, layer := range model.GetLayers() {
		outName := fmt.Sprintf("tensor_%d", i+1)

		switch l := layer.(type) {
		case *nn.Linear:
			wName := fmt.Sprintf("W_%d", nodeIdx)
			bName := fmt.Sprintf("B_%d", nodeIdx)

			wShape := l.Weight.Data.Shape() // [out, in]
			g.addInitializer(wName, wShape, l.Weight.Data.Data())
			if l.Bias != nil {
				bShape := l.Bias.Data.Shape()
				g.addInitializer(bName, bShape, l.Bias.Data.Data())
				g.addNode("Gemm", []string{cur, wName, bName}, []string{outName},
					fmt.Sprintf("gemm_%d", nodeIdx),
					[]onnxAttr{
						{name: "transB", intVal: 1, isInt: true},
					})
			} else {
				g.addNode("Gemm", []string{cur, wName}, []string{outName},
					fmt.Sprintf("gemm_%d", nodeIdx),
					[]onnxAttr{
						{name: "transB", intVal: 1, isInt: true},
					})
			}

		case *nn.ReLULayer:
			_ = l
			g.addNode("Relu", []string{cur}, []string{outName},
				fmt.Sprintf("relu_%d", nodeIdx), nil)

		case *nn.SigmoidLayer:
			_ = l
			g.addNode("Sigmoid", []string{cur}, []string{outName},
				fmt.Sprintf("sigmoid_%d", nodeIdx), nil)

		case *nn.TanhLayer:
			_ = l
			g.addNode("Tanh", []string{cur}, []string{outName},
				fmt.Sprintf("tanh_%d", nodeIdx), nil)

		case *nn.GELULayer:
			_ = l
			// GELU is approximated with ONNX Gelu op (opset 20) or built from erf.
			g.addNode("Gelu", []string{cur}, []string{outName},
				fmt.Sprintf("gelu_%d", nodeIdx), nil)

		case *nn.Dropout:
			_ = l
			// In inference mode Dropout is identity; map to Identity op.
			g.addNode("Identity", []string{cur}, []string{outName},
				fmt.Sprintf("drop_%d", nodeIdx), nil)

		default:
			return fmt.Errorf("onnx: unsupported layer type %T at index %d", layer, i)
		}

		cur = outName
		nodeIdx++
	}

	g.outputName = cur

	// ── Serialise ─────────────────────────────────────────────────────────────
	modelBytes := buildONNXModel(g, inputShape)
	return os.WriteFile(path, modelBytes, 0o644)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal graph representation
// ─────────────────────────────────────────────────────────────────────────────

type onnxNode struct {
	opType string
	inputs []string
	outs   []string
	name   string
	attrs  []onnxAttr
}

type onnxAttr struct {
	name   string
	intVal int64
	fVal   float32
	isInt  bool
	isFlt  bool
}

type onnxInitializer struct {
	name  string
	shape []int
	data  []float64
}

type onnxGraph struct {
	nodes        []onnxNode
	initializers []onnxInitializer
	inputName    string
	outputName   string
}

func (g *onnxGraph) addNode(opType string, inputs, outputs []string, name string, attrs []onnxAttr) {
	g.nodes = append(g.nodes, onnxNode{opType: opType, inputs: inputs, outs: outputs, name: name, attrs: attrs})
}

func (g *onnxGraph) addInitializer(name string, shape []int, data []float64) {
	g.initializers = append(g.initializers, onnxInitializer{name: name, shape: shape, data: data})
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal protobuf encoder
// ─────────────────────────────────────────────────────────────────────────────
//
// Wire types:
//   0  varint
//   2  length-delimited  (string, bytes, sub-message)
//   5  32-bit fixed      (float32)

type pbBuf []byte

func (b *pbBuf) appendVarint(v uint64) {
	for v >= 0x80 {
		*b = append(*b, byte(v)|0x80)
		v >>= 7
	}
	*b = append(*b, byte(v))
}

func (b *pbBuf) appendTag(field int, wireType int) {
	b.appendVarint(uint64(field<<3 | wireType))
}

// field + varint value
func (b *pbBuf) addInt(field int, v int64) {
	b.appendTag(field, 0)
	b.appendVarint(uint64(v))
}

// field + length-delimited bytes
func (b *pbBuf) addBytes(field int, data []byte) {
	b.appendTag(field, 2)
	b.appendVarint(uint64(len(data)))
	*b = append(*b, data...)
}

// field + UTF-8 string
func (b *pbBuf) addString(field int, s string) {
	b.addBytes(field, []byte(s))
}

// field + embedded sub-message
func (b *pbBuf) addMsg(field int, sub pbBuf) {
	b.addBytes(field, sub)
}

// float32 as fixed32
func (b *pbBuf) appendFloat32(v float32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
	*b = append(*b, buf[:]...)
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX serialisation
// ─────────────────────────────────────────────────────────────────────────────

func buildONNXModel(g *onnxGraph, inputShape []int) []byte {
	// GraphProto
	var graph pbBuf

	// 1: repeated NodeProto node
	for _, n := range g.nodes {
		graph.addMsg(1, buildNode(n))
	}
	// 2: string name
	graph.addString(2, "gotorch_graph")
	// 5: repeated TensorProto initializer
	for _, ini := range g.initializers {
		graph.addMsg(5, buildInitializer(ini))
	}
	// 11: repeated ValueInfoProto input
	graph.addMsg(11, buildValueInfo(g.inputName, inputShape))
	// 12: repeated ValueInfoProto output
	graph.addMsg(12, buildValueInfoDynamic(g.outputName))

	// ModelProto
	var model pbBuf
	model.addInt(1, 8) // ir_version = 8
	// 7: repeated OperatorSetIdProto opset_import
	var opset pbBuf
	opset.addString(1, "") // domain = "" (standard)
	opset.addInt(2, 17)    // version = 17
	model.addMsg(7, opset)
	// 8: GraphProto graph
	model.addMsg(8, graph)

	return []byte(model)
}

// buildNode serialises a NodeProto.
func buildNode(n onnxNode) pbBuf {
	var b pbBuf
	for _, inp := range n.inputs {
		b.addString(1, inp) // repeated string input
	}
	for _, out := range n.outs {
		b.addString(2, out) // repeated string output
	}
	b.addString(3, n.name)   // string name
	b.addString(4, n.opType) // string op_type
	for _, attr := range n.attrs {
		b.addMsg(5, buildAttr(attr))
	}
	return b
}

// buildAttr serialises an AttributeProto.
func buildAttr(a onnxAttr) pbBuf {
	var b pbBuf
	b.addString(1, a.name) // string name
	if a.isInt {
		b.addInt(4, a.intVal) // int64 i
		b.addInt(20, 2)       // AttributeType INT = 2
	} else if a.isFlt {
		b.appendTag(4, 5) // float f (fixed32)
		b.appendFloat32(a.fVal)
		b.addInt(20, 1) // AttributeType FLOAT = 1
	}
	return b
}

// buildInitializer serialises a TensorProto for a weight/bias tensor.
func buildInitializer(ini onnxInitializer) pbBuf {
	var b pbBuf
	// 1: repeated int64 dims
	for _, d := range ini.shape {
		b.addInt(1, int64(d))
	}
	// 2: int32 data_type (1 = FLOAT)
	b.addInt(2, 1)
	// 8: string name
	b.addString(8, ini.name)
	// 4: repeated float float_data  (field 4, wire type 2 for packed)
	rawFloats := make([]byte, len(ini.data)*4)
	for i, v := range ini.data {
		binary.LittleEndian.PutUint32(rawFloats[i*4:], math.Float32bits(float32(v)))
	}
	b.addBytes(4, rawFloats)
	return b
}

// buildValueInfo serialises a ValueInfoProto with a known shape.
func buildValueInfo(name string, shape []int) pbBuf {
	var b pbBuf
	b.addString(1, name)
	b.addMsg(2, buildTypeProto(shape))
	return b
}

// buildValueInfoDynamic serialises a ValueInfoProto with unknown shape.
func buildValueInfoDynamic(name string) pbBuf {
	var b pbBuf
	b.addString(1, name)
	// Emit an empty type (dynamic shape) — valid ONNX.
	var tp pbBuf
	b.addMsg(2, tp)
	return b
}

// buildTypeProto serialises a TypeProto for a float tensor.
func buildTypeProto(shape []int) pbBuf {
	// TypeProto_Tensor
	var tt pbBuf
	tt.addInt(1, 1) // elem_type = FLOAT
	tt.addMsg(2, buildTensorShape(shape))

	var tp pbBuf
	tp.addMsg(1, tt) // tensor_type
	return tp
}

// buildTensorShape serialises a TensorShapeProto.
func buildTensorShape(shape []int) pbBuf {
	var b pbBuf
	for i, d := range shape {
		var dim pbBuf
		if i == 0 {
			// dim 0 = batch (dynamic)
			dim.addString(2, "batch_size") // dim_param
		} else {
			dim.addInt(1, int64(d)) // dim_value
		}
		b.addMsg(1, dim)
	}
	return b
}

