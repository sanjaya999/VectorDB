package store

import (
	"container/heap"
	"sort"
)

type DistanceFunc func(a, b []float32) float32

// searchSort implements a simple linear search that returns the k closest results.
func (s *Store) searchSort(query []float32, k int, distanceFunc DistanceFunc) []Result {

	if len(query) != s.dim {
		panic("query dimension mismatch")
	}
	results := []Result{}

	for id := 0; id <= s.Len()-1; id++ {
		currentVector := s.Get(id)
		distance := distanceFunc(query, currentVector)
		// Store the result in a slice of Result structs
		result := Result{
			ID:   id,
			Dist: distance,
		}
		results = append(results, result)

	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Dist < results[j].Dist
	})
	if k > len(results) {
		k = len(results)
	}
	if k <= 0 {
		return []Result{}
	}
	return results[:k]
}

// heap search implementation
func (s *Store) Search(query []float32, k int, distanceFunc DistanceFunc) []Result {
	if len(query) != s.dim {
		panic("query dimension mismatch")
	}
	if k <= 0 {
		return []Result{}
	}
	if k > s.Len() {
		k = s.Len()
	}
	h := &resultHeap{}

	for id := 0; id <= s.Len()-1; id++ {
		currentVector := s.Get(id)
		distance := distanceFunc(query, currentVector)
		if h.Len() < k {
			heap.Push(h, Result{ID: id, Dist: distance})
		} else if distance < (*h)[0].Dist {
			heap.Pop(h)
			heap.Push(h, Result{ID: id, Dist: distance})
		}
	}
	results := make([]Result, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(Result)
	}
	return results
}
