package store

import (
	"slices"
	"testing"

	"github.com/sanjaya999/vectorDB/internal/distance"
)

// newSearchStore builds a small store used by most search tests.
// Distances (L2Squared) from query [0, 0]:

func newSearchStore() *Store {
	s := New(2)
	s.Add([]float32{0, 0})
	s.Add([]float32{10, 10})
	s.Add([]float32{1, 1})
	s.Add([]float32{3, 4})
	return s
}

// ids pulls just the IDs out of a result list, to make comparisons easy.
func ids(rs []Result) []int {
	out := make([]int, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

// Results come back closest first, with the right distances.
func TestSearchOrderAndDistances(t *testing.T) {
	s := newSearchStore()
	got := s.Search([]float32{0, 0}, 4, distance.L2Squared)

	wantIDs := []int{0, 2, 3, 1}
	wantDists := []float32{0, 2, 25, 200}

	if !slices.Equal(ids(got), wantIDs) {
		t.Fatalf("IDs = %v, want %v", ids(got), wantIDs)
	}
	for i, r := range got {
		if r.Dist != wantDists[i] {
			t.Errorf("result %d: Dist = %v, want %v", i, r.Dist, wantDists[i])
		}
	}
}

// Different values of k, including edge cases.
func TestSearchK(t *testing.T) {
	tests := []struct {
		name string
		k    int
		want []int
	}{
		{"k=1", 1, []int{0}},
		{"k=2", 2, []int{0, 2}},
		{"k equals Len", 4, []int{0, 2, 3, 1}},
		{"k bigger than Len", 10, []int{0, 2, 3, 1}},
		{"k=0", 0, []int{}},
		{"k negative", -1, []int{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newSearchStore()
			got := s.Search([]float32{0, 0}, tt.k, distance.L2Squared)
			if !slices.Equal(ids(got), tt.want) {
				t.Errorf("IDs = %v, want %v", ids(got), tt.want)
			}
		})
	}
}

// A different query point should give a different order.
// Distances from [10, 9]: id0=181, id1=1, id2=145, id3=74
func TestSearchDifferentQuery(t *testing.T) {
	s := newSearchStore()
	got := s.Search([]float32{10, 9}, 2, distance.L2Squared)
	want := []int{1, 3}
	if !slices.Equal(ids(got), want) {
		t.Errorf("IDs = %v, want %v", ids(got), want)
	}
}

// Searching an empty store returns nothing, without crashing.
func TestSearchEmptyStore(t *testing.T) {
	s := New(2)
	got := s.Search([]float32{0, 0}, 3, distance.L2Squared)
	if len(got) != 0 {
		t.Errorf("got %v, want no results", got)
	}
}

// A query with the wrong dimension must panic.
func TestSearchDimMismatchPanics(t *testing.T) {
	s := newSearchStore()
	mustPanic(t, "Search wrong dim", func() {
		s.Search([]float32{1, 2, 3}, 1, distance.L2Squared)
	})
}

// Search uses whatever distance function it's given.
// Negating L2 makes the farthest vector "closest".
func TestSearchCustomDistance(t *testing.T) {
	s := newSearchStore()
	farthest := func(a, b []float32) float32 {
		return -distance.L2Squared(a, b)
	}
	got := s.Search([]float32{0, 0}, 1, farthest)
	if len(got) != 1 || got[0].ID != 1 {
		t.Errorf("got %v, want ID 1 (the farthest vector)", ids(got))
	}
}

// Searching must not change the stored data.
func TestSearchDoesNotModifyStore(t *testing.T) {
	s := newSearchStore()
	s.Search([]float32{0, 0}, 4, distance.L2Squared)

	if s.Len() != 4 {
		t.Errorf("Len = %d after search, want 4", s.Len())
	}
	if v := s.Get(3); v[0] != 3 || v[1] != 4 {
		t.Errorf("vector 3 = %v after search, want [3 4]", v)
	}
}
