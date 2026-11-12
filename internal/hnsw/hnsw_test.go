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
		if got := idx.addNode(0); got != want {
			t.Fatalf("addNode() = %d, want %d", got, want)
		}
	}
	if idx.entry != 0 {
		t.Errorf("entry = %d, want 0", idx.entry)
	}

	idx.connect(0, 1, 0)
	idx.connect(1, 2, 0)
	idx.connect(0, 1, 0) // duplicate: must not be added twice
	idx.connect(2, 2, 0) // self-link: must be ignored

	want := [][]int{{1}, {0, 2}, {1}}
	for id, w := range want {
		if got := idx.neighbors[id][0]; !slices.Equal(got, w) {
			t.Errorf("neighbors[%d] = %v, want %v", id, got, w)
		}
	}
}

// buildGraph makes 1-D points and links them by hand on layer 0.
func buildGraph(points []float32, links [][2]int) *Index {
	s := store.New(1)
	idx := New(s, distance.L2Squared, 16)
	for _, p := range points {
		s.Add([]float32{p})
		idx.addNode(0)
	}
	for _, l := range links {
		idx.connect(l[0], l[1], 0)
	}
	return idx
}

func TestGreedyWalksToNearest(t *testing.T) {
	// chain: 0 - 1 - 2 - 3 - 4
	idx := buildGraph(
		[]float32{0, 1, 2, 3, 4},
		[][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}},
	)
	id, d := idx.greedy([]float32{3.2}, 0, 0)
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
	id, _ := idx.greedy([]float32{6}, 0, 0)
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
	got := idx.searchLayer(q, 0, 1, 0)
	if len(got) != 1 || got[0].ID != 1 {
		t.Errorf("ef=1: got %v, want only node 1", got)
	}

	// ef=2 keeps node 2 as a backup and finds node 3 through it.
	got = idx.searchLayer(q, 0, 2, 0)
	if len(got) != 2 || got[0].ID != 3 || got[1].ID != 1 {
		t.Errorf("ef=2: got %v, want node 3 then node 1", got)
	}
}

func TestSearchLayerChain(t *testing.T) {
	idx := buildGraph(
		[]float32{0, 1, 2, 3, 4},
		[][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}},
	)
	got := idx.searchLayer([]float32{3.2}, 0, 3, 0)
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
	// node 3 (value 10) links only to node 2: nodes 1 and 0 lie in the
	// same direction and can be reached through node 2
	if got := idx.neighbors[3][0]; !slices.Equal(got, []int{2}) {
		t.Errorf("neighbors[3] = %v, want [2]", got)
	}
	// node 2 links back, so node 3 can be reached
	if !slices.Contains(idx.neighbors[2][0], 3) {
		t.Errorf("node 2 should link back to node 3")
	}
}

func TestInsertThenSearch(t *testing.T) {
	idx := New(store.New(1), distance.L2Squared, 2)
	for v := range 10 {
		idx.Insert([]float32{float32(v)}) // values 0 through 9
	}
	got := idx.searchLayer([]float32{6.2}, idx.entry, 3, 0)
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
	idx.m = 1 // the cap is 2 × m, so 2 links

	idx.prune(0, 0)
	if got := idx.neighbors[0][0]; !slices.Equal(got, []int{1, 2}) {
		t.Errorf("neighbors[0] = %v, want [1 2]", got)
	}

	// node 1 has one link, under the cap, so nothing changes
	idx.prune(1, 0)
	if got := idx.neighbors[1][0]; !slices.Equal(got, []int{0}) {
		t.Errorf("neighbors[1] = %v, want [0]", got)
	}
}

func TestSelectNeighbors(t *testing.T) {
	// The node choosing neighbors sits at value 0.
	// id:     0   1   2    3
	// value:  1   2   -1   5
	idx := buildGraph([]float32{1, 2, -1, 5}, nil)
	cands := []store.Result{ // closest-first, distances measured from value 0
		{ID: 0, Dist: 1},
		{ID: 2, Dist: 1},
		{ID: 1, Dist: 4},
		{ID: 3, Dist: 25},
	}

	// node 0 (value 1) is picked. node 2 (value -1) is on the other side, so it's kept.
	// node 1 (value 2) and node 3 (value 5) are closer to node 0 than to us, so they're skipped.
	if got := idx.selectNeighbors(cands, 3); !slices.Equal(got, []int{0, 2}) {
		t.Errorf("got %v, want [0 2]", got)
	}

	// the limit still applies
	if got := idx.selectNeighbors(cands, 1); !slices.Equal(got, []int{0}) {
		t.Errorf("limit 1: got %v, want [0]", got)
	}
}

func TestSearch(t *testing.T) {
	empty := New(store.New(1), distance.L2Squared, 4)
	if got := empty.Search([]float32{1}, 3, 10); len(got) != 0 {
		t.Errorf("empty index: got %d results, want 0", len(got))
	}

	idx := New(store.New(1), distance.L2Squared, 4)
	for v := range 20 {
		idx.Insert([]float32{float32(v)}) // values 0 through 19
	}

	got := idx.Search([]float32{12.3}, 3, 10)
	want := []int{12, 13, 11}
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d", len(got), len(want))
	}
	for i, id := range want {
		if got[i].ID != id {
			t.Errorf("result %d: node %d, want node %d", i, got[i].ID, id)
		}
	}

	// ef smaller than k still returns k results
	if got := idx.Search([]float32{12.3}, 3, 1); len(got) != 3 {
		t.Errorf("ef < k: got %d results, want 3", len(got))
	}
}
