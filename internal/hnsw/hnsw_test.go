package hnsw

import (
	"math"
	"slices"
	"testing"

	"github.com/sanjaya999/vectorDB/internal/distance"
	"github.com/sanjaya999/vectorDB/internal/store"
)

func TestConnect(t *testing.T) {
	idx := New(store.New(2), distance.L2Squared, 16)

	if idx.entry != -1 {
		t.Fatalf("empty index: entry = %d, want -1", idx.entry)
	}
	for want := range 3 {
		if got := idx.addNode(); got != want {
			t.Fatalf("addNode() = %d, want %d", got, want)
		}
	}
	if idx.entry != 0 {
		t.Errorf("entry = %d, want 0", idx.entry)
	}

	idx.connect(0, 1)
	idx.connect(1, 2)
	idx.connect(0, 1) // duplicate: must not be added twice
	idx.connect(2, 2) // self-link: must be ignored

	want := [][]int{{1}, {0, 2}, {1}}
	for id, w := range want {
		if got := idx.neighbors[id]; !slices.Equal(got, w) {
			t.Errorf("neighbors[%d] = %v, want %v", id, got, w)
		}
	}
}

// buildGraph makes 1-D points and links them by hand.
func buildGraph(points []float32, links [][2]int) *Index {
	s := store.New(1)
	idx := New(s, distance.L2Squared, 16)
	for _, p := range points {
		s.Add([]float32{p})
		idx.addNode()
	}
	for _, l := range links {
		idx.connect(l[0], l[1])
	}
	return idx
}

func TestGreedyWalksToNearest(t *testing.T) {
	// chain: 0 - 1 - 2 - 3 - 4
	idx := buildGraph(
		[]float32{0, 1, 2, 3, 4},
		[][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}},
	)
	id, d := idx.greedy([]float32{3.2}, 0)
	if id != 3 {
		t.Errorf("id = %d, want 3", id)
	}
	if math.Abs(float64(d)-0.04) > 1e-4 {
		t.Errorf("dist = %v, want ~0.04", d)
	}
}

func TestGreedyGetsStuck(t *testing.T) {
	// id:     0   1   2    3
	// value:  0   3   10   6
	// links:  0-1, 0-2, 2-3
	idx := buildGraph(
		[]float32{0, 3, 10, 6},
		[][2]int{{0, 1}, {0, 2}, {2, 3}},
	)
	id, _ := idx.greedy([]float32{6}, 0)
	if id != 1 {
		t.Errorf("id = %d, want 1 (the dead end)", id)
	}
}

func TestSearchLayerEscapesDeadEnd(t *testing.T) {
	idx := buildGraph(
		[]float32{0, 3, 10, 6},
		[][2]int{{0, 1}, {0, 2}, {2, 3}},
	)
	q := []float32{6}

	// ef=1 has no room for a backup, so it gets stuck like greedy.
	got := idx.searchLayer(q, 0, 1)
	if len(got) != 1 || got[0].ID != 1 {
		t.Errorf("ef=1: got %v, want only node 1", got)
	}

	// ef=2 keeps node 2 as a backup and finds node 3 through it.
	got = idx.searchLayer(q, 0, 2)
	if len(got) != 2 || got[0].ID != 3 || got[1].ID != 1 {
		t.Errorf("ef=2: got %v, want node 3 then node 1", got)
	}
}

func TestSearchLayerChain(t *testing.T) {
	idx := buildGraph(
		[]float32{0, 1, 2, 3, 4},
		[][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}},
	)
	got := idx.searchLayer([]float32{3.2}, 0, 3)
	want := []int{3, 4, 2} // closest three to 3.2, in order
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d", len(got), len(want))
	}
	for i, id := range want {
		if got[i].ID != id {
			t.Errorf("result %d: node %d, want node %d", i, got[i].ID, id)
		}
	}
}
