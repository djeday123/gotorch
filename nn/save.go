package nn

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// checkpoint is the JSON-serializable format for model weights.
type checkpoint struct {
	Params []paramEntry `json:"params"`
}

type paramEntry struct {
	Shape []int     `json:"shape"`
	Data  []float64 `json:"data"`
}

// Save writes all model parameters to a JSON file at path.
// Usage: nn.Save(model.Parameters(), "model.json")
func Save(params []*autograd.Variable, path string) error {
	ck := checkpoint{Params: make([]paramEntry, len(params))}
	for i, p := range params {
		ck.Params[i] = paramEntry{
			Shape: p.Data.Shape(),
			Data:  p.Data.Data(),
		}
	}
	b, err := json.Marshal(ck)
	if err != nil {
		return fmt.Errorf("nn.Save: marshal: %w", err)
	}
	return os.WriteFile(path, b, 0644)
}

// Load reads weights from a JSON file and applies them to params in order.
// The number of params and their shapes must match the checkpoint.
// Usage: nn.Load(model.Parameters(), "model.json")
func Load(params []*autograd.Variable, path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("nn.Load: read: %w", err)
	}
	var ck checkpoint
	if err := json.Unmarshal(b, &ck); err != nil {
		return fmt.Errorf("nn.Load: unmarshal: %w", err)
	}
	if len(ck.Params) != len(params) {
		return fmt.Errorf("nn.Load: checkpoint has %d params, model has %d",
			len(ck.Params), len(params))
	}
	for i, entry := range ck.Params {
		p := params[i]
		modelShape := p.Data.Shape()
		if !shapeEqual(entry.Shape, modelShape) {
			return fmt.Errorf("nn.Load: param %d shape mismatch: checkpoint %v vs model %v",
				i, entry.Shape, modelShape)
		}
		p.Data = tensor.New(entry.Data, entry.Shape)
	}
	return nil
}

func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
