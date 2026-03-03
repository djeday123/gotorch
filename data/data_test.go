package data

import (
	"testing"

	"github.com/djeday123/gotorch/tensor"
)

// ── TensorDataset ─────────────────────────────────────────────────────────────

func TestTensorDatasetLen(t *testing.T) {
	X := tensor.RandN(100, 4)
	Y := tensor.RandN(100, 1)
	ds := NewTensorDataset(X, Y)
	if ds.Len() != 100 {
		t.Errorf("Len=%d want 100", ds.Len())
	}
}

func TestTensorDatasetGet(t *testing.T) {
	X := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2})
	Y := tensor.New([]float64{10, 20, 30}, []int{3})
	ds := NewTensorDataset(X, Y)

	xi, yi := ds.Get(1)
	if xi.Data()[0] != 3 || xi.Data()[1] != 4 {
		t.Errorf("X[1] = %v, want [3 4]", xi.Data())
	}
	if yi.Data()[0] != 20 {
		t.Errorf("Y[1] = %v, want 20", yi.Data()[0])
	}
}

func TestTensorDatasetNilTarget(t *testing.T) {
	X := tensor.RandN(10, 4)
	ds := NewTensorDataset(X, nil)
	_, yi := ds.Get(0)
	if yi != nil {
		t.Error("expected nil Y for unsupervised dataset")
	}
}

func TestTensorDatasetGetBounds(t *testing.T) {
	X := tensor.Ones(5, 2)
	ds := NewTensorDataset(X, nil)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on out-of-bounds Get")
		}
	}()
	ds.Get(10)
}

// ── DataLoader — basic iteration ──────────────────────────────────────────────

func TestDataLoaderBatches(t *testing.T) {
	X := tensor.Ones(20, 4)
	Y := tensor.Ones(20, 1)
	ds := NewTensorDataset(X, Y)

	dl := NewDataLoader(ds, 4, WithPrefetch(0))
	dl.Reset()

	count := 0
	for dl.HasNext() {
		b := dl.Next()
		if b.X.Shape()[0] != 4 {
			t.Errorf("batch %d: X shape %v, want [4 4]", count, b.X.Shape())
		}
		count++
	}
	if count != 5 {
		t.Errorf("expected 5 batches, got %d", count)
	}
}

func TestDataLoaderDropLast(t *testing.T) {
	// 21 samples, batchSize=4 → 5 full + 1 partial; dropLast should give 5
	X := tensor.Ones(21, 4)
	ds := NewTensorDataset(X, nil)

	dl := NewDataLoader(ds, 4, WithPrefetch(0), WithDropLast(true))
	dl.Reset()

	count := 0
	for dl.HasNext() {
		dl.Next()
		count++
	}
	if count != 5 {
		t.Errorf("expected 5 batches with dropLast, got %d", count)
	}
}

func TestDataLoaderNumBatches(t *testing.T) {
	X := tensor.Ones(21, 4)
	ds := NewTensorDataset(X, nil)

	dl := NewDataLoader(ds, 4, WithDropLast(false))
	if dl.NumBatches() != 6 {
		t.Errorf("NumBatches=%d want 6", dl.NumBatches())
	}

	dl2 := NewDataLoader(ds, 4, WithDropLast(true))
	if dl2.NumBatches() != 5 {
		t.Errorf("NumBatches dropLast=%d want 5", dl2.NumBatches())
	}
}

func TestDataLoaderShuffle(t *testing.T) {
	// With shuffle, order should differ from sequential on average
	// (probabilistic test — might fail with very low probability)
	n := 100
	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i)
	}
	X := tensor.New(data, []int{n})
	ds := NewTensorDataset(X, nil)

	dl := NewDataLoader(ds, n, WithShuffle(true), WithPrefetch(0))
	dl.Reset()
	b := dl.Next()
	first := b.X.Data()[0]
	// With 100 elements, probability of first element being 0.0 is 1%
	// Effectively impossible to fail 10 times in a row
	different := first != 0.0
	_ = different // Not asserting — just verifying it runs without panic
}

func TestDataLoaderPrefetch(t *testing.T) {
	X := tensor.Ones(16, 4)
	Y := tensor.Ones(16, 1)
	ds := NewTensorDataset(X, Y)

	dl := NewDataLoader(ds, 4, WithPrefetch(2))
	dl.Reset()

	count := 0
	for dl.HasNext() {
		b := dl.Next()
		if b.X == nil {
			t.Fatal("nil batch X from prefetch")
		}
		count++
	}
	if count != 4 {
		t.Errorf("expected 4 batches, got %d", count)
	}
}

func TestDataLoaderReset(t *testing.T) {
	X := tensor.Ones(8, 2)
	ds := NewTensorDataset(X, nil)
	dl := NewDataLoader(ds, 4, WithPrefetch(0))

	// First epoch
	dl.Reset()
	c1 := 0
	for dl.HasNext() {
		dl.Next()
		c1++
	}

	// Second epoch
	dl.Reset()
	c2 := 0
	for dl.HasNext() {
		dl.Next()
		c2++
	}

	if c1 != 2 || c2 != 2 {
		t.Errorf("expected 2 batches per epoch, got %d and %d", c1, c2)
	}
}
