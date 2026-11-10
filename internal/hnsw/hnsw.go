package hnsw

import (
	"container/heap"
	"slices"

	"github.com/sanjaya999/vectorDB/internal/store"
)

type Index struct {
	store     *store.Store       //vector lives in this store
	dist      store.DistanceFunc //measure closeness
	m         int                //max number of neighbors per node
	neighbors [][]int            // neighbors[id] = IDs that node `id` links to
	entry     int                // entry point to the graph (the first node added)
}

func New(s *store.Store, dist store.DistanceFunc, m int) *Index {
	return &Index{
		store:     s,
		dist:      dist,
		m:         m,
		neighbors: make([][]int, 0, s.Len()),
		entry:     -1,
	}
}

func (idx *Index) addNode() int {
	id := len(idx.neighbors)
	idx.neighbors = append(idx.neighbors, make([]int, 0, idx.m))

	if idx.entry == -1 {
		idx.entry = id
	}

	return id
}

func (idx *Index) connect(a, b int) {
	if a >= len(idx.neighbors) || b >= len(idx.neighbors) {
		panic("invalid node ID")
	}
	if a == b {
		return
	}
	if slices.Contains(idx.neighbors[a], b) {
		return
	}
	idx.neighbors[a] = append(idx.neighbors[a], b)
	idx.neighbors[b] = append(idx.neighbors[b], a)
}

func (idx *Index) greedy(query []float32, start int) (int, float32) {
	current := start
	currentDistance := idx.dist(query, idx.store.Get(current))

	for {
		prev := current
		for _, vec := range idx.neighbors[current] {
			dist := idx.dist(query, idx.store.Get(vec))
			if dist < currentDistance {
				current = vec
				currentDistance = dist
			}
		}
		if current == prev {
			return current, currentDistance
		}
	}
}

// searchLayer explores the graph from start and returns the ef closest
// nodes it finds, sorted closest-first.
func (idx *Index) searchLayer(query []float32, start int, ef int) []store.Result {
	first := store.Result{ID: start, Dist: idx.dist(query, idx.store.Get(start))}

	visited := map[int]bool{start: true}
	candidates := &minHeap{first} // to explore, closest on top
	results := &maxHeap{first}    // best so far, worst on top

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(store.Result)
		if c.Dist > (*results)[0].Dist {
			break
		}
		for _, neighbor := range idx.neighbors[c.ID] {
			if visited[neighbor] {
				continue
			}
			d := idx.dist(query, idx.store.Get(neighbor))
			visited[neighbor] = true

			if len(*results) < ef || d < (*results)[0].Dist {
				r := store.Result{ID: neighbor, Dist: d}
				heap.Push(results, r)
				heap.Push(candidates, r)

				if len(*results) > ef {
					heap.Pop(results)
				}

			}
		}
	}

	// Empty the max-heap from the back so the slice ends up closest-first.
	out := make([]store.Result, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(results).(store.Result)
	}
	return out
}
