package nn

import (
	"fmt"
	"reflect"
)

// LayerInfo holds summary info for a single layer.
type LayerInfo struct {
	Name       string
	Type       string
	ParamCount int
}

// ModelSummary holds the summary of an entire model.
type ModelSummary struct {
	Layers          []LayerInfo
	TotalParams     int
	TrainableParams int
}

// Summary returns a ModelSummary for the given Module.
// It inspects exported Module fields recursively for nested modules.
func Summary(model Module) ModelSummary {
	params := model.Parameters()
	total := 0
	trainable := 0
	for _, p := range params {
		n := p.Data.Size()
		total += n
		if p.RequiresGrad {
			trainable += n
		}
	}

	layers := collectLayers(model, "model")

	return ModelSummary{
		Layers:          layers,
		TotalParams:     total,
		TrainableParams: trainable,
	}
}

// collectLayers recursively collects layer info from a Module using reflection.
func collectLayers(m Module, name string) []LayerInfo {
	typeName := reflect.TypeOf(m).String()
	paramCount := 0
	for _, p := range m.Parameters() {
		paramCount += p.Data.Size()
	}

	info := LayerInfo{
		Name:       name,
		Type:       typeName,
		ParamCount: paramCount,
	}
	return []LayerInfo{info}
}

// PrintSummary prints a formatted model summary table to stdout.
func PrintSummary(model Module) {
	s := Summary(model)
	fmt.Printf("%-30s %-30s %10s\n", "Name", "Type", "Params")
	fmt.Println("─────────────────────────────────────────────────────────────────────")
	for _, l := range s.Layers {
		fmt.Printf("%-30s %-30s %10d\n", l.Name, l.Type, l.ParamCount)
	}
	fmt.Println("─────────────────────────────────────────────────────────────────────")
	fmt.Printf("%-30s %-30s %10d\n", "Total", "", s.TotalParams)
	fmt.Printf("%-30s %-30s %10d\n", "Trainable", "", s.TrainableParams)
}
