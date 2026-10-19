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



## License

MIT