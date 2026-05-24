package data

import (
	"math/rand"
	"sync"

	"github.com/djeday123/gotorch/tensor"
)

// Batch is a single mini-batch returned by DataLoader.
type Batch struct {
	X *tensor.Tensor // [batchSize, ...]
	Y *tensor.Tensor // [batchSize, ...] or nil
}

// DataLoader iterates over a Dataset in mini-batches.
// Call Reset() before each epoch, then loop HasNext() / Next().
//
// Concurrency:
//   - All public methods are safe to call from a single consumer goroutine.
//   - Internally, a prefetch worker may run concurrently; it produces batches
//     by index (not by shared cursor), so no shared mutable iteration state
//     is touched.
//   - Reset()/HasNext()/Next() are serialised via `mu` so that calling Reset()
//     while a consumer is between HasNext() and Next() is well-defined.
type DataLoader struct {
	dataset   Dataset
	batchSize int
	shuffle   bool
	dropLast  bool
	prefetch  int // number of batches to prefetch

	mu           sync.Mutex
	indices      []int
	totalBatches int        // set in Reset()
	consumed     int        // batches consumed so far
	ch           chan Batch // buffer of pre-loaded batches (prefetch>0)
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
	dl.mu.Lock()
	// Stop any running prefetch goroutine before we mutate iteration state.
	if dl.done != nil {
		close(dl.done)
		oldCh := dl.ch
		dl.done = nil
		dl.ch = nil
		dl.mu.Unlock()
		// Drain remaining items so the worker can exit on its send path.
		for range oldCh {
		}
		dl.mu.Lock()
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
	dl.consumed = 0
	dl.totalBatches = dl.numBatchesLocked()

	if dl.prefetch > 0 {
		dl.ch = make(chan Batch, dl.prefetch)
		dl.done = make(chan struct{})
		ch := dl.ch
		done := dl.done
		total := dl.totalBatches
		indices := dl.indices
		dl.mu.Unlock()
		go dl.prefetchWorker(ch, done, total, indices)
		return
	}
	dl.mu.Unlock()
}

// HasNext reports whether there are more batches available this epoch.
func (dl *DataLoader) HasNext() bool {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	return dl.consumed < dl.totalBatches
}

// Next returns the next batch. Blocks if prefetch is not yet ready.
func (dl *DataLoader) Next() Batch {
	dl.mu.Lock()
	if dl.consumed >= dl.totalBatches {
		dl.mu.Unlock()
		return Batch{}
	}
	idx := dl.consumed
	dl.consumed++
	if dl.prefetch > 0 {
		ch := dl.ch
		dl.mu.Unlock()
		b, ok := <-ch
		if !ok {
			return Batch{}
		}
		return b
	}
	indices := dl.indices
	dl.mu.Unlock()
	return dl.loadBatchAt(idx, indices)
}

// NumBatches returns the total number of batches per epoch.
func (dl *DataLoader) NumBatches() int {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	return dl.numBatchesLocked()
}

func (dl *DataLoader) numBatchesLocked() int {
	n := dl.dataset.Len()
	if dl.dropLast {
		return n / dl.batchSize
	}
	return (n + dl.batchSize - 1) / dl.batchSize
}

// prefetchWorker loads all batches into the channel asynchronously. It owns
// the channel close (via defer) and reads its inputs by value so the main
// goroutine can safely replace dl.ch / dl.done under the mutex.
func (dl *DataLoader) prefetchWorker(ch chan<- Batch, done <-chan struct{}, total int, indices []int) {
	defer close(ch)
	for i := 0; i < total; i++ {
		batch := dl.loadBatchAt(i, indices)
		select {
		case <-done:
			return
		case ch <- batch:
		}
	}
}

// loadBatchAt is a pure function of (batchIdx, indices): no shared state.
// Safe to call concurrently from any goroutine.
func (dl *DataLoader) loadBatchAt(batchIdx int, indices []int) Batch {
	n := len(indices)
	start := batchIdx * dl.batchSize
	end := start + dl.batchSize
	if start > n {
		start = n
	}
	if end > n {
		end = n
	}
	idxs := indices[start:end]

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
