package hnsw

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
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
