package hnsw

import (
	"container/heap"
	"math/rand"
	"slices"
	"sort"

	"github.com/sanjaya999/vectorDB/internal/store"
)

type Index struct {
	store          *store.Store       //vector lives in this store
	dist           store.DistanceFunc //measure closeness
	m              int                //max number of neighbors per node
	neighbors      [][][]int          // neighbors[id] = IDs that node `id` links to
	efConstruction int                //size of the dynamic candidate list during construction
	entry          int                // entry point to the graph (the first node added)
	maxLevel       int                // maximum level of the graph (not used in this simplified version)
	rng            *rand.Rand         //random number generator for reproducibility
}

// Store returns the store that holds the index's vectors.
func (idx *Index) Store() *store.Store {
	return idx.store
}
func New(s *store.Store, dist store.DistanceFunc, m int) *Index {
	return &Index{
		store:          s,
		dist:           dist,
		m:              m,
		neighbors:      make([][][]int, 0, s.Len()),
		efConstruction: 100,
		entry:          -1,
		maxLevel:       0,
		rng:            rand.New(rand.NewSource(42)), // fixed seed for reproducibility
	}
}

// randomLevel picks the top layer for a new node. Each extra level has a
// 1-in-m chance, so most nodes stay on layer 0 and very few go high.
func (idx *Index) randomLevel() int {
	level := 0
	for idx.rng.Float64() < 1/float64(idx.m) {
		level++
	}
	return level
}

func (idx *Index) addNode(level int) int {
	id := len(idx.neighbors)
	idx.neighbors = append(idx.neighbors, make([][]int, level+1))

	if idx.entry == -1 {
		idx.entry = id
	}

	return id
}

func (idx *Index) connect(a, b, layer int) {
	if a >= len(idx.neighbors) || b >= len(idx.neighbors) {
		panic("invalid node ID")
	}
	if a == b {
		return
	}
	if slices.Contains(idx.neighbors[a][layer], b) {
		return
	}
	idx.neighbors[a][layer] = append(idx.neighbors[a][layer], b)
	idx.neighbors[b][layer] = append(idx.neighbors[b][layer], a)
}

func (idx *Index) greedy(query []float32, start, layer int) (int, float32) {
	current := start
	currentDistance := idx.dist(query, idx.store.Get(current))

	for {
		prev := current
		for _, vec := range idx.neighbors[current][layer] {
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
func (idx *Index) searchLayer(query []float32, start int, ef, layer int) []store.Result {
	first := store.Result{ID: start, Dist: idx.dist(query, idx.store.Get(start))}

	visited := map[int]bool{start: true}
	candidates := &minHeap{first} // to explore, closest on top
	results := &maxHeap{first}    // best so far, worst on top

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(store.Result)
		if c.Dist > (*results)[0].Dist {
			break
		}
		for _, neighbor := range idx.neighbors[c.ID][layer] {
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

func (idx *Index) Insert(v []float32) int {
	id := idx.store.Add(v)
	level := idx.randomLevel()
	idx.addNode(level)
	if id == 0 {
		idx.maxLevel = level
		return id
	}
	ep := idx.entry
	for layer := idx.maxLevel; layer > level+1; layer-- {
		ep, _ = idx.greedy(v, ep, layer)
	}
	for layer := min(level, idx.maxLevel); layer >= 0; layer-- {
		// Search for the nearest existing nodes.
		found := idx.searchLayer(v, ep, idx.efConstruction, layer)

		for _, n := range idx.selectNeighbors(found, idx.m) {
			idx.connect(id, n, layer)
			idx.prune(n, layer)
		}
	}
	if level > idx.maxLevel {
		idx.maxLevel = level
		idx.entry = id
	}
	return id
}

func (idx *Index) prune(id, layer int) {
	if id >= len(idx.neighbors) {
		panic("invalid node ID")
	}
	maxLinks := idx.m
	if layer == 0 {
		maxLinks = 2 * idx.m
	}
	if len(idx.neighbors[id][layer]) <= maxLinks {
		return
	}
	v := idx.store.Get(id)
	links := idx.neighbors[id][layer]
	cands := make([]store.Result, len(links))
	for i, n := range links {
		cands[i] = store.Result{ID: n, Dist: idx.dist(v, idx.store.Get(n))}
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].Dist < cands[j].Dist })
	idx.neighbors[id][layer] = idx.selectNeighbors(cands, maxLinks)
}

func (idx *Index) Search(query []float32, k, ef int) []store.Result {
	if idx.entry == -1 || k <= 0 {
		return []store.Result{}
	}
	if ef < k {
		ef = k
	}
	ep := idx.entry
	for layer := idx.maxLevel; layer > 0; layer-- {
		ep, _ = idx.greedy(query, ep, layer)
	}
	results := idx.searchLayer(query, ep, ef, 0)
	return results[0:min(k, len(results))]
}

func (idx *Index) selectNeighbors(candidates []store.Result, limit int) []int {
	var selected []int
	for _, candidate := range candidates {
		if len(selected) >= limit {
			return selected
		}

		if len(selected) == 0 {
			selected = append(selected, candidate.ID)
			continue
		}
		v := idx.store.Get(candidate.ID)
		isGood := true
		for _, s := range selected {
			w := idx.store.Get(s)
			if idx.dist(v, w) < candidate.Dist {
				isGood = false
				break
			}
		}
		if isGood {
			selected = append(selected, candidate.ID)
		}
	}
	return selected
}

// AvgLinks returns the average number of links per node.
func (idx *Index) AvgLinks() float64 {
	total := 0
	for _, layers := range idx.neighbors {
		total += len(layers[0])
	}
	return float64(total) / float64(len(idx.neighbors))
}
