package store

type resultHeap []Result

// heap implementation
func (h resultHeap) Len() int {
	return len(h)
}

func (h resultHeap) Less(i, j int) bool {
	return h[i].Dist > h[j].Dist
}

func (h resultHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *resultHeap) Push(x any) {
	*h = append(*h, x.(Result))
}

func (h *resultHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
