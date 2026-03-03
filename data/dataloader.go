package data

import (
	"math/rand"

	"github.com/djeday123/gotorch/tensor"
)

// Batch is a single mini-batch returned by DataLoader.
type Batch struct {
	X *tensor.Tensor // [batchSize, ...]
	Y *tensor.Tensor // [batchSize, ...] or nil
}

// DataLoader iterates over a Dataset in mini-batches.
// Call Reset() before each epoch, then loop HasNext() / Next().
type DataLoader struct {
	dataset   Dataset
	batchSize int
	shuffle   bool
	dropLast  bool
	prefetch  int // number of batches to prefetch

	indices      []int
	cursor       int   // position in indices (used only in sync path or by worker)
	totalBatches int   // set in Reset()
	consumed     int   // batches consumed so far (main goroutine)
	ch           chan Batch
	done         chan struct{}
}

// DataLoaderOption configures a DataLoader.
type DataLoaderOption func(*DataLoader)

// WithShuffle enables shuffling at each epoch.
func WithShuffle(shuffle bool) DataLoaderOption {
	return func(d *DataLoader) { d.shuffle = shuffle }
}

// WithDropLast drops the last partial batch.
func WithDropLast(drop bool) DataLoaderOption {
	return func(d *DataLoader) { d.dropLast = drop }
}

// WithPrefetch sets the number of batches to prefetch asynchronously.
// 0 = synchronous. Default: 2.
func WithPrefetch(n int) DataLoaderOption {
	return func(d *DataLoader) { d.prefetch = n }
}

// NewDataLoader creates a DataLoader over dataset.
func NewDataLoader(dataset Dataset, batchSize int, opts ...DataLoaderOption) *DataLoader {
	dl := &DataLoader{
		dataset:   dataset,
		batchSize: batchSize,
		prefetch:  2,
	}
	for _, o := range opts {
		o(dl)
	}
	return dl
}

// Reset resets to the start of a new epoch (reshuffles if needed).
// Must be called before the first use and between epochs.
func (dl *DataLoader) Reset() {
	// Stop any running prefetch goroutine
	if dl.done != nil {
		close(dl.done)
		for range dl.ch { // drain
		}
		dl.done = nil
	}

	n := dl.dataset.Len()
	dl.indices = make([]int, n)
	for i := range dl.indices {
		dl.indices[i] = i
	}
	if dl.shuffle {
		rand.Shuffle(n, func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}
	dl.cursor = 0
	dl.consumed = 0
	dl.totalBatches = dl.NumBatches()

	if dl.prefetch > 0 {
		dl.ch = make(chan Batch, dl.prefetch)
		dl.done = make(chan struct{})
		go dl.prefetchWorker()
	}
}

// HasNext reports whether there are more batches available this epoch.
func (dl *DataLoader) HasNext() bool {
	return dl.consumed < dl.totalBatches
}

// Next returns the next batch. Blocks if prefetch is not yet ready.
func (dl *DataLoader) Next() Batch {
	dl.consumed++
	if dl.prefetch > 0 {
		return <-dl.ch
	}
	return dl.loadBatch()
}

// NumBatches returns the total number of batches per epoch.
func (dl *DataLoader) NumBatches() int {
	n := dl.dataset.Len()
	if dl.dropLast {
		return n / dl.batchSize
	}
	return (n + dl.batchSize - 1) / dl.batchSize
}

// prefetchWorker loads all batches into the channel asynchronously.
func (dl *DataLoader) prefetchWorker() {
	defer close(dl.ch)
	total := dl.totalBatches
	for i := 0; i < total; i++ {
		batch := dl.loadBatch()
		select {
		case <-dl.done:
			return
		case dl.ch <- batch:
		}
	}
}

// loadBatch reads the next batch from dl.indices starting at dl.cursor.
// NOTE: must not be called concurrently — cursor is not protected.
func (dl *DataLoader) loadBatch() Batch {
	n := dl.dataset.Len()
	end := dl.cursor + dl.batchSize
	if end > n {
		end = n
	}
	idxs := dl.indices[dl.cursor:end]
	dl.cursor = end

	var xRows, yRows []*tensor.Tensor
	for _, i := range idxs {
		xi, yi := dl.dataset.Get(i)
		xRows = append(xRows, xi)
		if yi != nil {
			yRows = append(yRows, yi)
		}
	}

	xBatch := stackRows(xRows)
	var yBatch *tensor.Tensor
	if len(yRows) > 0 {
		yBatch = stackRows(yRows)
	}
	return Batch{X: xBatch, Y: yBatch}
}

// stackRows stacks same-shape tensors along a new leading dimension.
func stackRows(rows []*tensor.Tensor) *tensor.Tensor {
	if len(rows) == 0 {
		return tensor.Zeros(0)
	}
	rowShape := rows[0].Shape()
	rowSize := rows[0].Size()

	batchShape := make([]int, len(rowShape)+1)
	batchShape[0] = len(rows)
	copy(batchShape[1:], rowShape)

	data := make([]float64, len(rows)*rowSize)
	for i, r := range rows {
		copy(data[i*rowSize:], r.Data())
	}
	return tensor.New(data, batchShape)
}
