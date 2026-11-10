package dataset

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// readRaw reads .fvecs/.ivecs records as raw uint32 words, flattened.
// limit <= 0 reads everything.
func readRaw(path string, limit int) ([]uint32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	var out []uint32
	dim := -1
	hdr := make([]byte, 4)
	var buf []byte

	for n := 0; limit <= 0 || n < limit; n++ {
		if _, err := io.ReadFull(r, hdr); err != nil {
			if err == io.EOF {
				break
			}
			return nil, 0, err
		}
		d := int(binary.LittleEndian.Uint32(hdr))
		if dim == -1 {
			dim = d
			buf = make([]byte, 4*d)
		} else if d != dim {
			return nil, 0, fmt.Errorf("record %d: dim %d, want %d", n, d, dim)
		}
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, 0, err
		}
		for i := 0; i < d; i++ {
			out = append(out, binary.LittleEndian.Uint32(buf[4*i:]))
		}
	}
	return out, dim, nil
}

// ReadFvecs returns all vectors flattened (vector i at [i*dim:(i+1)*dim]) and dim.
func ReadFvecs(path string, limit int) ([]float32, int, error) {
	raw, dim, err := readRaw(path, limit)
	if err != nil {
		return nil, 0, err
	}
	out := make([]float32, len(raw))
	for i, w := range raw {
		out[i] = math.Float32frombits(w)
	}
	return out, dim, nil
}

// ReadIvecs returns one []int per record (e.g. ground-truth neighbor IDs).
func ReadIvecs(path string, limit int) ([][]int, error) {
	raw, dim, err := readRaw(path, limit)
	if err != nil {
		return nil, err
	}
	out := make([][]int, len(raw)/dim)
	for i := range out {
		row := make([]int, dim)
		for j := range row {
			row[j] = int(int32(raw[i*dim+j]))
		}
		out[i] = row
	}
	return out, nil
}
