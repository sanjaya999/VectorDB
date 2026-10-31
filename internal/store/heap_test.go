package store

import (
	"container/heap"
	"testing"
)

func TestResultHeapPopsLargestFirst(t *testing.T) {
	h := &resultHeap{}

	for i, d := range []float32{5, 1, 9, 3} {
		heap.Push(h, Result{ID: i, Dist: d})
	}

	want := []float32{9, 5, 3, 1}
	for _, w := range want {
		got := heap.Pop(h).(Result)
		if got.Dist != w {
			t.Errorf("popped %v, want %v", got.Dist, w)
		}
	}

	if h.Len() != 0 {
		t.Errorf("heap should be empty, has %d items", h.Len())
	}
}
