package store

import "sort"

type DistanceFunc func(a, b []float32) float32

func (s *Store) Search(query []float32, k int, distanceFunc DistanceFunc) []Result {

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
