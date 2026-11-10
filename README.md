# vectorDB

A vector database written from  in Go.

## Project layout

```
internal/distance/   distance functions and tests
internal/store/      flat vector storage and search
```

## Design notes

- **Flat storage:** all vectors live in one `[]float32`, and vector `i` is
  `data[i*dim : (i+1)*dim]`. This means one allocation and good cache locality.
- **Squared L2:** the square root is skipped, since it doesn't change the
  ranking of neighbors and costs time.
- **`float32` everywhere:** it matches what embedding models and standard
  datasets use, and halves memory compared to `float64`.


## Run the tests

```bash
go test -race ./...
```

## Benchmark

Benchmark results from i5 11th gen 8GB RAM System:

```
inserted 100000 (58s)
inserted 200000 (2m18s)
inserted 300000 (3m45s)
inserted 400000 (5m27s)
inserted 500000 (7m3s)
inserted 600000 (8m42s)
inserted 700000 (10m16s)
inserted 800000 (13m7s)
inserted 900000 (14m46s)
inserted 1000000 (16m28s)
built index: 1000000 vectors, m=16, in 16m28s

exact        recall@10 = 1.0000 |     30.5 QPS
hnsw ef=10   recall@10 = 0.7460 |   7917.5 QPS
hnsw ef=20   recall@10 = 0.8595 |   6221.9 QPS
hnsw ef=50   recall@10 = 0.9490 |   4013.5 QPS
hnsw ef=100  recall@10 = 0.9860 |   2793.7 QPS
hnsw ef=200  recall@10 = 0.9960 |   1606.9 QPS
```



## License

MIT