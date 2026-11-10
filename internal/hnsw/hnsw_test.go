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

func TestInsert(t *testing.T) {
	idx := New(store.New(1), distance.L2Squared, 2)
	for want, v := range []float32{0, 1, 2, 10} {
		if got := idx.Insert([]float32{v}); got != want {
			t.Fatalf("Insert returned id %d, want %d", got, want)
		}
	}
	// node 3 (value 10) links to its two nearest: node 2, then node 1
	if got := idx.neighbors[3]; !slices.Equal(got, []int{2, 1}) {
		t.Errorf("neighbors[3] = %v, want [2 1]", got)
	}
	// links go both ways
	if !slices.Contains(idx.neighbors[2], 3) || !slices.Contains(idx.neighbors[1], 3) {
		t.Errorf("nodes 2 and 1 should link back to node 3")
	}
}

func TestInsertThenSearch(t *testing.T) {
	idx := New(store.New(1), distance.L2Squared, 2)
	for v := range 10 {
		idx.Insert([]float32{float32(v)}) // values 0 through 9
	}
	got := idx.searchLayer([]float32{6.2}, idx.entry, 3)
	want := []int{6, 7, 5} // the three values closest to 6.2
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d", len(got), len(want))
	}
	for i, id := range want {
		if got[i].ID != id {
			t.Errorf("result %d: node %d, want node %d", i, got[i].ID, id)
		}
	}
}

func TestPrune(t *testing.T) {
	// node 0 (value 5) linked to nodes at values 4, 6, and 20
	idx := buildGraph([]float32{5, 4, 6, 20}, [][2]int{{0, 1}, {0, 2}, {0, 3}})
	idx.m = 2

	idx.prune(0)
	if got := idx.neighbors[0]; !slices.Equal(got, []int{1, 2}) {
		t.Errorf("neighbors[0] = %v, want [1 2]", got)
	}

	// node 1 has one link, under the cap, so nothing changes
	idx.prune(1)
	if got := idx.neighbors[1]; !slices.Equal(got, []int{0}) {
		t.Errorf("neighbors[1] = %v, want [0]", got)
	}
}
