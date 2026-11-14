package hnsw

import (
	"bytes"
	"encoding/binary"
	"slices"
	"testing"
)

func TestWriteHeader(t *testing.T) {
	h := fileHeader{
		Magic: fileMagic, Version: formatVersion,
		Dim: 128, Count: 1_000_000, M: 16, EfConstruction: 100,
		Entry: 0x1234, MaxLevel: 4,
	}
	var buf bytes.Buffer // an in-memory stand-in for a file
	if err := writeHeader(&buf, h); err != nil {
		t.Fatalf("writeHeader: %v", err)
	}

	want := []byte{
		'V', 'D', 'B', '1', // magic
		0x01, 0, 0, 0, // version 1
		0x80, 0, 0, 0, // dim 128
		0x40, 0x42, 0x0f, 0, // count 1,000,000
		0x10, 0, 0, 0, // m 16
		0x64, 0, 0, 0, // efConstruction 100
		0x34, 0x12, 0, 0, // entry 0x1234
		0x04, 0, 0, 0, // maxLevel 4
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("header bytes:\n got % x\nwant % x", buf.Bytes(), want)
	}
}

func TestHeaderRoundTrip(t *testing.T) {
	want := fileHeader{
		Magic: fileMagic, Version: formatVersion,
		Dim: 128, Count: 1_000_000, M: 16, EfConstruction: 100,
		Entry: 0x1234, MaxLevel: 4,
	}
	var buf bytes.Buffer
	if err := writeHeader(&buf, want); err != nil {
		t.Fatalf("writeHeader: %v", err)
	}
	got, err := readHeader(&buf)
	if err != nil {
		t.Fatalf("readHeader: %v", err)
	}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestHeaderEmptyIndex(t *testing.T) {
	want := fileHeader{Magic: fileMagic, Version: formatVersion, Dim: 8, M: 16, EfConstruction: 100, Entry: -1}
	var buf bytes.Buffer
	if err := writeHeader(&buf, want); err != nil {
		t.Fatalf("writeHeader: %v", err)
	}
	if e := buf.Bytes()[24:28]; !bytes.Equal(e, []byte{0xff, 0xff, 0xff, 0xff}) {
		t.Errorf("entry bytes = % x, want ff ff ff ff", e)
	}
	got, err := readHeader(&buf)
	if err != nil {
		t.Fatalf("readHeader: %v", err)
	}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestReadHeaderRejectsBadFiles(t *testing.T) {
	valid := fileHeader{Magic: fileMagic, Version: formatVersion, Dim: 4, Entry: -1}

	encode := func(h fileHeader) []byte {
		var buf bytes.Buffer
		if err := writeHeader(&buf, h); err != nil {
			t.Fatalf("writeHeader: %v", err)
		}
		return buf.Bytes()
	}

	badMagic := encode(valid)
	badMagic[0] = 'X'

	newer := valid
	newer.Version = formatVersion + 1

	cases := []struct {
		name string
		data []byte
	}{
		{"wrong magic", badMagic},
		{"newer version", encode(newer)},
		{"truncated", encode(valid)[:10]},
		{"empty file", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h, err := readHeader(bytes.NewReader(c.data))
			if err == nil {
				t.Fatalf("got header %+v, want an error", h)
			}
			t.Logf("error: %v", err)
		})
	}
}
func TestWriteVectorsBytes(t *testing.T) {
	var buf bytes.Buffer
	if err := writeVectors(&buf, []float32{1, 2}); err != nil {
		t.Fatalf("writeVectors: %v", err)
	}
	want := []byte{
		0x00, 0x00, 0x80, 0x3f, // 1.0
		0x00, 0x00, 0x00, 0x40, // 2.0
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("got % x, want % x", buf.Bytes(), want)
	}
}

func TestWriteVectorsAcrossChunks(t *testing.T) {
	data := make([]float32, 3*vectorChunk+5) // three full chunks and a short one
	for i := range data {
		data[i] = float32(i)
	}
	var buf bytes.Buffer
	if err := writeVectors(&buf, data); err != nil {
		t.Fatalf("writeVectors: %v", err)
	}
	if buf.Len() != 4*len(data) {
		t.Fatalf("wrote %d bytes, want %d", buf.Len(), 4*len(data))
	}
	got := make([]float32, len(data))
	if err := binary.Read(&buf, binary.LittleEndian, got); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if !slices.Equal(got, data) {
		t.Fatal("data changed between write and read")
	}
}

func TestWriteVectorsEmpty(t *testing.T) {
	var buf bytes.Buffer
	if err := writeVectors(&buf, nil); err != nil {
		t.Fatalf("writeVectors: %v", err)
	}
	if buf.Len() != 0 {
		t.Fatalf("wrote %d bytes for an empty store, want 0", buf.Len())
	}
}

func TestVectorsRoundTrip(t *testing.T) {
	data := make([]float32, 3*vectorChunk+5)
	for i := range data {
		data[i] = float32(i) * 0.5
	}
	var buf bytes.Buffer
	if err := writeVectors(&buf, data); err != nil {
		t.Fatalf("writeVectors: %v", err)
	}
	got, err := readVectors(&buf, len(data))
	if err != nil {
		t.Fatalf("readVectors: %v", err)
	}
	if !slices.Equal(got, data) {
		t.Fatal("data changed between write and read")
	}
}

func TestReadVectorsTruncated(t *testing.T) {
	var buf bytes.Buffer
	if err := writeVectors(&buf, make([]float32, 10)); err != nil {
		t.Fatalf("writeVectors: %v", err)
	}
	short := buf.Bytes()[:4*10-3] // last float is missing 3 bytes
	if _, err := readVectors(bytes.NewReader(short), 10); err == nil {
		t.Fatal("want an error for a truncated vector section")
	}
}

func TestReadVectorsEmpty(t *testing.T) {
	got, err := readVectors(bytes.NewReader(nil), 0)
	if err != nil {
		t.Fatalf("readVectors: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("got %d floats, want 0", len(got))
	}
}

// u32s encodes numbers the way the graph section stores them.
func u32s(vals ...uint32) []byte {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, vals)
	return buf.Bytes()
}

func sameGraph(a, b [][][]int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for l := range a[i] {
			if !slices.Equal(a[i][l], b[i][l]) {
				return false
			}
		}
	}
	return true
}

var testGraph = [][][]int{
	{{1, 2}},      // node 0
	{{0, 2}, nil}, // node 1: two layers, the top one has no links
	{{0, 1}},      // node 2
}

func TestWriteGraphBytes(t *testing.T) {
	var buf bytes.Buffer
	if err := writeGraph(&buf, testGraph); err != nil {
		t.Fatalf("writeGraph: %v", err)
	}
	want := u32s(
		1, 2, 1, 2,
		2, 2, 0, 2, 0,
		1, 2, 0, 1,
	)
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("graph bytes:\n got % x\nwant % x", buf.Bytes(), want)
	}
}

func TestGraphRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := writeGraph(&buf, testGraph); err != nil {
		t.Fatalf("writeGraph: %v", err)
	}
	got, err := readGraph(&buf, len(testGraph))
	if err != nil {
		t.Fatalf("readGraph: %v", err)
	}
	if !sameGraph(got, testGraph) {
		t.Fatalf("got %v, want %v", got, testGraph)
	}
}

func TestReadGraphRejectsBadData(t *testing.T) {
	var valid bytes.Buffer
	if err := writeGraph(&valid, testGraph); err != nil {
		t.Fatalf("writeGraph: %v", err)
	}

	cases := []struct {
		name  string
		count int
		data  []byte
	}{
		{"link to a missing node", 2, u32s(1, 1, 5, 1, 0)},
		{"node with zero layers", 1, u32s(0)},
		{"too many layers", 1, u32s(1000)},
		{"more links than nodes", 1, u32s(1, 9)},
		{"truncated", 3, valid.Bytes()[:valid.Len()-2]},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g, err := readGraph(bytes.NewReader(c.data), c.count)
			if err == nil {
				t.Fatalf("got graph %v, want an error", g)
			}
			t.Logf("error: %v", err)
		})
	}
}
