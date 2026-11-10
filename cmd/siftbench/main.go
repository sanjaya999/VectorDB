package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/sanjaya999/vectorDB/internal/dataset"
	"github.com/sanjaya999/vectorDB/internal/distance"
	"github.com/sanjaya999/vectorDB/internal/store"
)

func main() {
	dir := flag.String("dir", "sift", "path to extracted SIFT folder")
	nBase := flag.Int("n", 0, "number of base vectors (0 = all 1M)")
	nq := flag.Int("q", 200, "number of queries to run")
	k := flag.Int("k", 10, "k for recall@k")
	flag.Parse()

	t0 := time.Now()
	base, dim, err := dataset.ReadFvecs(*dir+"/sift_base.fvecs", *nBase)
	if err != nil {
		log.Fatal(err)
	}
	queries, _, err := dataset.ReadFvecs(*dir+"/sift_query.fvecs", *nq)
	if err != nil {
		log.Fatal(err)
	}
	gt, err := dataset.ReadIvecs(*dir+"/sift_groundtruth.ivecs", *nq)
	if err != nil {
		log.Fatal(err)
	}

	s := store.New(dim)
	for i := 0; i < len(base)/dim; i++ {
		s.Add(base[i*dim : (i+1)*dim])
	}
	fmt.Printf("loaded %d base vectors, %d queries in %v\n", s.Len(), len(gt), time.Since(t0))

	results := make([][]store.Result, len(gt))
	t1 := time.Now()
	var wg sync.WaitGroup
	jobs := make(chan int)
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for qi := range jobs {
				q := queries[qi*dim : (qi+1)*dim]
				results[qi] = s.Search(q, *k, distance.L2Squared)
			}
		}()
	}
	for qi := range gt {
		jobs <- qi
	}
	close(jobs)
	wg.Wait()
	elapsed := time.Since(t1)

	hits := 0
	for qi, res := range results {
		q := queries[qi*dim : (qi+1)*dim]
		kth := distance.L2Squared(q, s.Get(gt[qi][*k-1])) // distance to the true k-th neighbor
		for _, r := range res {
			if r.Dist <= kth {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(len(gt)**k)
	fmt.Printf("recall@%d = %.4f | %d queries in %v | %.1f QPS\n",
		*k, recall, len(gt), elapsed, float64(len(gt))/elapsed.Seconds())
}
