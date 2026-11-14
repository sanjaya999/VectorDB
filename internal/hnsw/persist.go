package hnsw

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"

	"github.com/sanjaya999/vectorDB/internal/store"
)

const formatVersion = 1
const maxFileLayers = 64

var fileMagic = [4]byte{'V', 'D', 'B', '1'}

type fileHeader struct {
	Magic          [4]byte
	Version        uint32
	Dim            uint32
	Count          uint32
	M              uint32
	EfConstruction uint32
	Entry          int32
	MaxLevel       uint32
}

const vectorChunk = 1 << 16

func writeHeader(w io.Writer, h fileHeader) error {
	error := binary.Write(w, binary.LittleEndian, h)
	return error
}

func readHeader(r io.Reader) (fileHeader, error) {
	var h fileHeader
	if err := binary.Read(r, binary.LittleEndian, &h); err != nil {
		return fileHeader{}, fmt.Errorf("reading header: %w", err)
	}
	if h.Magic != fileMagic {
		return fileHeader{}, fmt.Errorf("not a VectorDB index file: starts with %q", h.Magic[:])
	}
	if h.Version != formatVersion {
		return fileHeader{}, fmt.Errorf("unsupported index format version %d (this code reads version %d)", h.Version, formatVersion)
	}
	return h, nil
}

func writeVectors(w io.Writer, data []float32) error {

	for start := 0; start < len(data); start += vectorChunk {
		end := min(start+vectorChunk, len(data))
		if err := binary.Write(w, binary.LittleEndian, data[start:end]); err != nil {
			return fmt.Errorf("writing vectors: %w", err)
		}
	}
	return nil
}

// readVectors reads n float32s from r.
func readVectors(r io.Reader, n int) ([]float32, error) {
	data := make([]float32, n)
	for start := 0; start < len(data); start += vectorChunk {
		end := min(start+vectorChunk, len(data))
		if err := binary.Read(r, binary.LittleEndian, data[start:end]); err != nil {
			return nil, fmt.Errorf("reading vectors: %w", err)
		}
	}
	return data, nil
}

func writeUint32(w io.Writer, v uint32) error {
	return binary.Write(w, binary.LittleEndian, v)
}

func readUint32(r io.Reader) (uint32, error) {
	var v uint32
	err := binary.Read(r, binary.LittleEndian, &v)
	return v, err
}

func writeGraph(w io.Writer, neighbors [][][]int) error {
	for _, layers := range neighbors {
		if err := writeUint32(w, uint32(len(layers))); err != nil {
			return fmt.Errorf("writing graph: %w", err)
		}
		for _, links := range layers {
			if err := writeUint32(w, uint32(len(links))); err != nil {
				return fmt.Errorf("writing graph: %w", err)
			}
			for _, id := range links {
				if err := writeUint32(w, uint32(id)); err != nil {
					return fmt.Errorf("writing graph: %w", err)
				}
			}
		}
	}
	return nil
}

func readGraph(r io.Reader, count int) ([][][]int, error) {
	neighbors := make([][][]int, count)
	for node := range neighbors {
		nLayers, err := readUint32(r)
		if err != nil {
			return nil, fmt.Errorf("reading graph: %w", err)
		}
		if nLayers < 1 || nLayers > maxFileLayers {
			return nil, fmt.Errorf("reading graph: invalid number of layers %d", nLayers)
		}
		layers := make([][]int, nLayers)
		for layer := range layers {
			nLinks, err := readUint32(r)
			if err != nil {
				return nil, fmt.Errorf("reading graph: %w", err)
			}
			if int(nLinks) > count {
				return nil, fmt.Errorf("reading graph: node %d layer %d has %d links, but the index has %d nodes", node, layer, nLinks, count)
			}
			links := make([]int, nLinks)
			for i := 0; i < int(nLinks); i++ {
				id, err := readUint32(r)
				if err != nil {
					return nil, fmt.Errorf("reading graph: %w", err)
				}
				if int(id) >= count {
					return nil, fmt.Errorf("reading graph: node %d links to node %d, but the index has %d nodes", node, id, count)
				}
				links[i] = int(id)
			}

			layers[layer] = links
		}
		neighbors[node] = layers
	}
	return neighbors, nil
}

func (idx *Index) Save(path string) error {
	if idx.store.Len() != len(idx.neighbors) {
		return fmt.Errorf("index has %d vectors but %d graph nodes", idx.store.Len(), len(idx.neighbors))
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1<<20)
	h := fileHeader{
		Magic:          fileMagic,
		Version:        formatVersion,
		Dim:            uint32(idx.store.Dim()),
		Count:          uint32(idx.store.Len()),
		M:              uint32(idx.m),
		EfConstruction: uint32(idx.efConstruction),
		Entry:          int32(idx.entry),
		MaxLevel:       uint32(idx.maxLevel),
	}
	if err := writeHeader(w, h); err != nil {
		return fmt.Errorf("writing header: %w", err)
	}
	if err := writeVectors(w, idx.store.Data()); err != nil {
		return err
	}
	if err := writeGraph(w, idx.neighbors); err != nil {
		return err
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flushing to %s: %w", path, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("closing %s: %w", path, err)
	}
	return nil
}

// checkHeader rejects header values that cannot describe a real index.
func checkHeader(h fileHeader, fileSize int64) error {
	if h.Dim < 1 || h.Dim > 1<<16 {
		return fmt.Errorf("invalid dimension %d", h.Dim)
	}
	if h.M < 2 {
		return fmt.Errorf("invalid m %d", h.M)
	}
	if h.EfConstruction < 1 {
		return fmt.Errorf("invalid efConstruction %d", h.EfConstruction)
	}
	if h.MaxLevel >= maxFileLayers {
		return fmt.Errorf("invalid max level %d", h.MaxLevel)
	}
	if h.Count == 0 && h.Entry != -1 {
		return fmt.Errorf("empty index has entry %d, want -1", h.Entry)
	}
	if h.Count > 0 && (h.Entry < 0 || int64(h.Entry) >= int64(h.Count)) {
		return fmt.Errorf("entry %d is not a valid node (index has %d nodes)", h.Entry, h.Count)
	}
	if need := 32 + 4*int64(h.Count)*int64(h.Dim); fileSize < need {
		return fmt.Errorf("file is %d bytes, too short for %d vectors of dimension %d", fileSize, h.Count, h.Dim)
	}
	return nil
}

// Load reads an index written by Save.
// dist must be the same distance function the index was built with.
func Load(path string, dist store.DistanceFunc) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	r := bufio.NewReaderSize(f, 1<<20)
	h, err := readHeader(r)
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	if err := checkHeader(h, info.Size()); err != nil {
		return nil, fmt.Errorf("checking header: %w", err)
	}
	data, err := readVectors(r, int(h.Count)*int(h.Dim))
	if err != nil {
		return nil, fmt.Errorf("reading vectors: %w", err)
	}
	s, err := store.FromData(int(h.Dim), data)
	if err != nil {
		return nil, fmt.Errorf("creating store: %w", err)
	}
	neighbors, err := readGraph(r, int(h.Count))
	if err != nil {
		return nil, fmt.Errorf("reading graph: %w", err)
	}
	if h.Count > 0 && len(neighbors[h.Entry]) != int(h.MaxLevel)+1 {
		return nil, fmt.Errorf("loading %s: entry node %d has %d layers, but max level is %d",
			path, h.Entry, len(neighbors[h.Entry]), h.MaxLevel)
	}
	return &Index{
		store:          s,
		neighbors:      neighbors,
		m:              int(h.M),
		efConstruction: int(h.EfConstruction),
		entry:          int(h.Entry),
		maxLevel:       int(h.MaxLevel),
		dist:           dist,
		rng:            rand.New(rand.NewSource(int64(h.Count) + 1)),
	}, nil
}
