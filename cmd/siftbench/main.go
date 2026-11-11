package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sanjaya999/vectorDB/internal/dataset"
	"github.com/sanjaya999/vectorDB/internal/distance"
	"github.com/sanjaya999/vectorDB/internal/hnsw"
	"github.com/sanjaya999/vectorDB/internal/store"
)

// runAll runs search for every query across all CPU cores and times it.
func runAll(nq int, search func(qi int) []store.Result) ([][]store.Result, time.Duration) {
	results := make([][]store.Result, nq)
	jobs := make(chan int)
	var wg sync.WaitGroup
	t := time.Now()
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for qi := range jobs {
				results[qi] = search(qi)
			}
		}()
	}
	for qi := 0; qi < nq; qi++ {
		jobs <- qi
	}
	close(jobs)
	wg.Wait()
	return results, time.Since(t)
}

func main() {
	dir := flag.String("dir", "sift", "path to extracted SIFT folder")
	nBase := flag.Int("n", 0, "number of base vectors (0 = all 1M)")
	nq := flag.Int("q", 200, "number of queries")
	k := flag.Int("k", 10, "k for recall@k")
	m := flag.Int("m", 16, "max links per node")
	efs := flag.String("ef", "10,20,50,100,200", "ef values to try, comma separated")
	flag.Parse()

	base, dim, err := dataset.ReadFvecs(*dir+"/sift_base.fvecs", *nBase)
	if err != nil {
		log.Fatal(err)
	}
	queries, _, err := dataset.ReadFvecs(*dir+"/sift_query.fvecs", *nq)
	if err != nil {
		log.Fatal(err)
	}
	n := len(base) / dim
	numQ := len(queries) / dim
	query := func(qi int) []float32 { return queries[qi*dim : (qi+1)*dim] }

	// Build: each Insert stores the vector and links it into the graph.
	s := store.New(dim)
	idx := hnsw.New(s, distance.L2Squared, *m)
	t0 := time.Now()
	for i := 0; i < n; i++ {
		idx.Insert(base[i*dim : (i+1)*dim])
		if (i+1)%100000 == 0 {
			fmt.Printf("  inserted %d (%v)\n", i+1, time.Since(t0).Round(time.Second))
		}
	}
	fmt.Printf("built index: %d vectors, m=%d, in %v\n\n", n, *m, time.Since(t0).Round(time.Second))
	fmt.Printf("average links per node: %.1f\n", idx.AvgLinks())
	// Exact search: the true answers and the baseline speed.
	truth, exactTime := runAll(numQ, func(qi int) []store.Result {
		return s.Search(query(qi), *k, distance.L2Squared)
	})
	fmt.Printf("exact        recall@%d = 1.0000 | %8.1f QPS\n", *k, float64(numQ)/exactTime.Seconds())

	// HNSW at each ef value.
	for _, f := range strings.Split(*efs, ",") {
		ef, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			log.Fatalf("bad ef value %q", f)
		}
		got, elapsed := runAll(numQ, func(qi int) []store.Result {
			return idx.Search(query(qi), *k, ef)
		})
		hits := 0
		for qi, res := range got {
			kth := truth[qi][len(truth[qi])-1].Dist // distance of the true k-th neighbor
			for _, r := range res {
				if r.Dist <= kth {
					hits++
				}
			}
		}
		recall := float64(hits) / float64(numQ*(*k))
		fmt.Printf("hnsw ef=%-4d recall@%d = %.4f | %8.1f QPS\n",
			ef, *k, recall, float64(numQ)/elapsed.Seconds())
	}
}
