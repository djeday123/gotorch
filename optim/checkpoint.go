package optim

import (
	"encoding/json"
	"fmt"
	"os"
)

// ---------------------------------------------------------------------------
// Optimizer checkpoint — save / load optimizer state to/from JSON
// ---------------------------------------------------------------------------

// SaveOptimizer saves the optimizer's internal state to path as JSON.
// Supports AdamW (and extendable to others).
func SaveOptimizer(opt Optimizer, path string) error {
	var state interface{}
	switch o := opt.(type) {
	case *AdamW:
		state = o.GetState()
	default:
		return fmt.Errorf("SaveOptimizer: unsupported optimizer type %T", opt)
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("SaveOptimizer marshal: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// LoadOptimizer restores optimizer state from a JSON file saved by SaveOptimizer.
func LoadOptimizer(opt Optimizer, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("LoadOptimizer read: %w", err)
	}

	switch o := opt.(type) {
	case *AdamW:
		var s AdamWState
		if err := json.Unmarshal(data, &s); err != nil {
			return fmt.Errorf("LoadOptimizer unmarshal: %w", err)
		}
		o.SetState(s)
	default:
		return fmt.Errorf("LoadOptimizer: unsupported optimizer type %T", opt)
	}
	return nil
}
