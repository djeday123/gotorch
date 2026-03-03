// Пример 4: LSTM Language Model (символьный уровень)
//
// Тренирует маленькую символьную языковую модель на строке с повторяющимся паттерном.
// Цель: предсказать следующий символ.
//
// Архитектура:
//   Embedding(vocabSize, 16) → LSTM(16→64) → Linear(64, vocabSize)
//
// Показывает:
//   - Embedding.Lookup() + LSTM в GoTorch
//   - CrossEntropyLoss для языкового моделирования
//   - Генерацию текста (greedy decoding)
package main

import (
	"fmt"
	"strings"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/nn"
	"github.com/djeday123/gotorch/optim"
	"github.com/djeday123/gotorch/tensor"
)

const (
	lmEmbDim  = 16
	lmHidden  = 64
	lmEpochs  = 200
	lmLR      = 0.005
)

// Обучающий текст с повторяющимися паттернами
var trainText = strings.Repeat("abcdefghij", 80) + strings.Repeat("hello world ", 40)

// CharLM — символьная языковая модель.
type CharLM struct {
	emb *nn.Embedding
	fc  *nn.Linear
	// LSTM — храним отдельно (не в Module slice, т.к. другой Forward)
	lstm     *nn.LSTM
	allParams []*autograd.Variable
}

func NewCharLM(vocabSize int) *CharLM {
	emb := nn.NewEmbedding(vocabSize, lmEmbDim)
	lstm := nn.NewLSTM(lmEmbDim, lmHidden)
	fc := nn.NewLinear(lmHidden, vocabSize, true)

	var params []*autograd.Variable
	params = append(params, emb.Parameters()...)
	params = append(params, lstm.Parameters()...)
	params = append(params, fc.Parameters()...)

	return &CharLM{
		emb:       emb,
		lstm:      lstm,
		fc:        fc,
		allParams: params,
	}
}

// Forward: токены [seqLen] → логиты [seqLen, vocabSize]
func (m *CharLM) Forward(tokens []int) *autograd.Variable {
	seqLen := len(tokens)

	// Embedding lookup: [seqLen, embDim]
	embOut := m.emb.Lookup(tokens)

	// LSTM: x [seqLen, embDim] → outputs [seqLen]*Variable([hiddenSize])
	outputs, _ := m.lstm.Forward(embOut, nil)

	// Применяем FC к каждому шагу, собираем логиты
	vocabSize := m.fc.Weight.Data.Shape()[0]
	logitData := make([]float64, seqLen*vocabSize)
	for t, h := range outputs {
		// h: [hiddenSize] → reshape to [1, hiddenSize]
		hVar := autograd.NewVar(h.Data.Reshape(1, lmHidden), h.RequiresGrad)
		step := m.fc.Forward(hVar) // [1, vocabSize]
		stepD := step.Data.Data()
		copy(logitData[t*vocabSize:], stepD)
	}

	// Для backprop: берём логиты через последний hidden → gradient приближение
	// (полный BPTT требует сохранения всего графа, здесь упрощённо)
	logitT := tensor.New(logitData, []int{seqLen, vocabSize})

	// Приближённый backward: используем последний выход для loss
	lastH := outputs[seqLen-1]
	lastHVar := autograd.NewVar(lastH.Data.Reshape(1, lmHidden), lastH.RequiresGrad)
	lastLogits := m.fc.Forward(lastHVar) // [1, vocabSize] — для градиента

	// Создаём result, который несёт полный tensor данных, но grad идёт через lastLogits
	return autograd.NewResult(logitT, &lstmLMBackward{lastLogits}, lastLogits)
}

type lstmLMBackward struct {
	logits *autograd.Variable
}

func (b *lstmLMBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// Упрощённо: передаём сумму gradients из всех строк
	gData := grad.Data()
	n := len(gData) / b.logits.Data.Size()
	out := make([]float64, b.logits.Data.Size())
	vocabSize := b.logits.Data.Size()
	for row := 0; row < n; row++ {
		for c := 0; c < vocabSize; c++ {
			out[c] += gData[row*vocabSize+c]
		}
	}
	return []*tensor.Tensor{tensor.New(out, b.logits.Data.Shape())}
}

func (m *CharLM) ZeroGrad() {
	for _, p := range m.allParams {
		p.ZeroGrad()
	}
}

func main() {
	// ── Словарь ──────────────────────────────────────────────────────────────
	charSet := make(map[rune]int)
	for _, ch := range trainText {
		charSet[ch] = 0
	}
	vocab := make([]rune, 0, len(charSet))
	for ch := range charSet {
		vocab = append(vocab, ch)
	}
	for i := 0; i < len(vocab)-1; i++ {
		for j := i + 1; j < len(vocab); j++ {
			if vocab[i] > vocab[j] {
				vocab[i], vocab[j] = vocab[j], vocab[i]
			}
		}
	}
	charToIdx := make(map[rune]int)
	idxToChar := make(map[int]rune)
	for i, ch := range vocab {
		charToIdx[ch] = i
		idxToChar[i] = ch
	}
	vocabSize := len(vocab)

	tokens := make([]int, len(trainText))
	for i, ch := range trainText {
		tokens[i] = charToIdx[ch]
	}

	fmt.Printf("Словарь: %d символов\n", vocabSize)
	fmt.Printf("Текст: %d токенов\n\n", len(tokens))

	model := NewCharLM(vocabSize)
	opt := optim.NewAdam(model.allParams, lmLR, 0.9, 0.999, 1e-8)

	totalP := 0
	for _, p := range model.allParams {
		totalP += p.Data.Size()
	}
	fmt.Printf("Параметров: %d\n\n", totalP)
	fmt.Printf("%-6s  %-10s\n", "Epoch", "Loss")
	fmt.Println("─────────────────")

	// ── Обучение ─────────────────────────────────────────────────────────────
	seqLen := 40
	for epoch := 1; epoch <= lmEpochs; epoch++ {
		totalLoss := 0.0
		steps := 0

		for start := 0; start+seqLen+1 < len(tokens) && steps < 15; start += seqLen {
			inp := tokens[start : start+seqLen]
			targets := tokens[start+1 : start+seqLen+1]

			model.ZeroGrad()
			logits := model.Forward(inp)
			loss := nn.CrossEntropyLoss(logits, targets)
			loss.Backward()
			optim.ClipGradNorm(model.allParams, 5.0)
			opt.Step()

			totalLoss += loss.Data.Item()
			steps++
		}

		if epoch%20 == 0 {
			fmt.Printf("%-6d  %.4f\n", epoch, totalLoss/float64(steps))
		}
	}

	// ── Генерация ────────────────────────────────────────────────────────────
	fmt.Println("\n── Генерация текста (greedy) ─────────────")
	for _, seedChar := range []rune{'a', 'h', ' '} {
		context := []int{charToIdx[seedChar]}
		generated := string(seedChar)

		for i := 0; i < 25; i++ {
			logits := model.Forward(context)
			seqL := logits.Data.Shape()[0]
			// Argmax последнего токена
			bestIdx, bestVal := 0, logits.Data.At(seqL-1, 0)
			for c := 1; c < vocabSize; c++ {
				if v := logits.Data.At(seqL-1, c); v > bestVal {
					bestVal = v
					bestIdx = c
				}
			}
			generated += string(idxToChar[bestIdx])
			context = append(context, bestIdx)
			if len(context) > 15 {
				context = context[1:]
			}
		}
		fmt.Printf("  %q → %q\n", string(seedChar), generated)
	}
	fmt.Println("\n✅ LSTM LM готова!")
}
