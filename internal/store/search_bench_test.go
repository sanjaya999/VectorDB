package store

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/sanjaya999/vectorDB/internal/distance"
)

// randomVector makes a vector of random numbers between 0 and 1.
func randomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

// randomStore fills a store with n random vectors.
func randomStore(n, dim int) *Store {
	s := New(dim)
	for range n {
		s.Add(randomVector(dim))
	}
	return s
}

func BenchmarkSearch(b *testing.B) {
	const dim = 128 // a typical small embedding size

	for _, n := range []int{1_000, 10_000, 100_000} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			s := randomStore(n, dim)
			query := randomVector(dim)

			b.ResetTimer() // don't count the time spent building the store
			for i := 0; i < b.N; i++ {
				s.Search(query, 10, distance.L2Squared)
			}
		})
	}
}
