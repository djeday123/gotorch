// Пример 5: Mini Transformer — классификация последовательностей
//
// Задача: определить "тональность" синтетической последовательности токенов.
// Позитивные: содержат токены {0,1,2} в начале.
// Негативные: содержат токены {3,4,5} в начале.
//
// Архитектура:
//   Embedding(vocabSize=8, d=32)
//   → 2× TransformerEncoderLayer(d=32, heads=4, ffn=64)
//   → Mean pooling по seq dim
//   → Linear(32 → 2)
//
// Показывает: Embedding, TransformerEncoderLayer, stacked layers, mean pooling.
package main

import (
	"fmt"
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/nn"
	"github.com/djeday123/gotorch/optim"
	"github.com/djeday123/gotorch/tensor"
)

const (
	tfVocab  = 8
	tfDim    = 32
	tfHeads  = 4
	tfFFN    = 64
	tfSeqLen = 10
	tfTrain  = 500
	tfTest   = 100
	tfEpochs = 30
	tfLR     = 0.001
	tfBatch  = 16
)

// TransformerClassifier: Embedding + 2 Transformer layers + mean pool + Linear.
type TransformerClassifier struct {
	emb    *nn.Embedding
	layer1 *nn.TransformerEncoderLayer
	layer2 *nn.TransformerEncoderLayer
	fc     *nn.Linear
}

func NewTransformerClassifier() *TransformerClassifier {
	return &TransformerClassifier{
		emb:    nn.NewEmbedding(tfVocab, tfDim),
		layer1: nn.NewTransformerEncoderLayer(tfDim, tfHeads, tfFFN, 0.1),
		layer2: nn.NewTransformerEncoderLayer(tfDim, tfHeads, tfFFN, 0.1),
		fc:     nn.NewLinear(tfDim, 2, true),
	}
}

func (m *TransformerClassifier) Parameters() []*autograd.Variable {
	var p []*autograd.Variable
	for _, mod := range []nn.Module{m.emb, m.layer1, m.layer2, m.fc} {
		p = append(p, mod.Parameters()...)
	}
	return p
}

func (m *TransformerClassifier) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}

// Forward: tokens []int → logits [2]
func (m *TransformerClassifier) Forward(tokens []int) *autograd.Variable {
	seqLen := len(tokens)

	// Embedding [seqLen, d]
	embOut := m.emb.Lookup(tokens) // [seqLen, tfDim]

	// 2 Transformer layers
	h := m.layer1.Forward(embOut)  // [seqLen, tfDim]
	h = m.layer2.Forward(h)        // [seqLen, tfDim]

	// Mean pooling по seq dim: [1, tfDim]
	pooled := meanPool(h, seqLen, tfDim)

	// Классификатор: [1, tfDim] → [1, 2]
	return m.fc.Forward(pooled)
}

// meanPool: усредняет [seqLen, d] → [1, d].
func meanPool(x *autograd.Variable, seqLen, d int) *autograd.Variable {
	data := x.Data.Data()
	out := make([]float64, d)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < d; j++ {
			out[j] += data[i*d+j]
		}
	}
	for j := range out {
		out[j] /= float64(seqLen)
	}
	// Простой backward: scale by 1/seqLen
	outT := tensor.New(out, []int{1, d})
	return autograd.NewResult(outT, &meanPoolBwd{x.Data, seqLen, d}, x)
}

type meanPoolBwd struct {
	xData  *tensor.Tensor
	seqLen int
	d      int
}

func (f *meanPoolBwd) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	g := grad.Data()
	scale := 1.0 / float64(f.seqLen)
	out := make([]float64, f.seqLen*f.d)
	for i := 0; i < f.seqLen; i++ {
		for j := 0; j < f.d; j++ {
			out[i*f.d+j] = g[j] * scale
		}
	}
	return []*tensor.Tensor{tensor.New(out, f.xData.Shape())}
}

// generateData: создаёт (tokens, label) пары.
// label=0 (негативный): начало с {3,4,5}
// label=1 (позитивный): начало с {0,1,2}
func generateData(n int) ([][]int, []int) {
	seqs := make([][]int, n)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		label := rand.Intn(2)
		labels[i] = label
		seq := make([]int, tfSeqLen)
		for j := range seq {
			if j < 3 {
				if label == 1 {
					seq[j] = rand.Intn(3) // {0,1,2}
				} else {
					seq[j] = 3 + rand.Intn(3) // {3,4,5}
				}
			} else {
				seq[j] = rand.Intn(tfVocab)
			}
		}
		seqs[i] = seq
	}
	return seqs, labels
}

func main() {
	rand.Seed(42)

	trainSeqs, trainLabels := generateData(tfTrain)
	testSeqs, testLabels := generateData(tfTest)

	fmt.Printf("Vocab: %d, seqLen: %d, d_model: %d, heads: %d\n",
		tfVocab, tfSeqLen, tfDim, tfHeads)
	fmt.Printf("Train: %d, Test: %d\n\n", tfTrain, tfTest)

	model := NewTransformerClassifier()
	opt := optim.NewAdam(model.Parameters(), tfLR, 0.9, 0.999, 1e-8)
	sched := optim.NewCosineAnnealingLR(opt, tfEpochs, 1e-6)

	totalP := 0
	for _, p := range model.Parameters() {
		totalP += p.Data.Size()
	}
	fmt.Printf("Параметров: %d\n\n", totalP)
	fmt.Printf("%-6s  %-10s  %-12s  %-10s\n", "Epoch", "Loss", "Train Acc", "Test Acc")
	fmt.Println("──────────────────────────────────────────────")

	for epoch := 1; epoch <= tfEpochs; epoch++ {
		perm := rand.Perm(tfTrain)
		totalLoss := 0.0
		steps := 0

		for b := 0; b < tfTrain; b += tfBatch {
			end := b + tfBatch
			if end > tfTrain {
				end = tfTrain
			}

			batchLoss := 0.0
			for _, idx := range perm[b:end] {
				model.ZeroGrad()
				logits := model.Forward(trainSeqs[idx]) // [1, 2]
				loss := nn.CrossEntropyLoss(logits, []int{trainLabels[idx]})
				loss.Backward()
				batchLoss += loss.Data.Item()
			}
			// Усредняем loss (уже накоплены grad от каждого примера)
			// Шаг оптимизатора после батча
			optim.ClipGradNorm(model.Parameters(), 1.0)
			opt.Step()

			totalLoss += batchLoss / float64(end-b)
			steps++
		}

		sched.Step()
		avgLoss := totalLoss / float64(steps)

		if epoch%5 == 0 || epoch == 1 {
			trainAcc := evalTransformer(model, trainSeqs[:100], trainLabels[:100])
			testAcc := evalTransformer(model, testSeqs, testLabels)
			fmt.Printf("%-6d  %-10.4f  %-12.2f%%  %-10.2f%%\n",
				epoch, avgLoss, trainAcc*100, testAcc*100)
		}
	}

	fmt.Println("\n✅ Transformer обучен!")

	// Пример предсказания
	fmt.Println("\nПримеры предсказаний:")
	for i := 0; i < 5; i++ {
		logits := model.Forward(testSeqs[i])
		class := 0
		if logits.Data.At(0, 1) > logits.Data.At(0, 0) {
			class = 1
		}
		correct := "✓"
		if class != testLabels[i] {
			correct = "✗"
		}
		fmt.Printf("  Seq: %v → pred=%d (true=%d) %s\n",
			testSeqs[i][:5], class, testLabels[i], correct)
	}
}

func evalTransformer(model *TransformerClassifier, seqs [][]int, labels []int) float64 {
	correct := 0
	for i, seq := range seqs {
		logits := model.Forward(seq)
		pred := 0
		if logits.Data.At(0, 1) > logits.Data.At(0, 0) {
			pred = 1
		}
		if pred == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(labels))
}
