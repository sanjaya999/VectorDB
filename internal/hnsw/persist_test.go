package hnsw

import (
	"bytes"
	"encoding/binary"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sanjaya999/vectorDB/internal/distance"
	"github.com/sanjaya999/vectorDB/internal/store"
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

// buildTestIndex makes a small index of random vectors.
func buildTestIndex(n, dim int) *Index {
	rng := rand.New(rand.NewSource(42))
	idx := New(store.New(dim), distance.L2Squared, 4)
	for range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()
		}
		idx.Insert(v)
	}
	return idx
}

func TestSaveWritesWholeFile(t *testing.T) {
	const n, dim = 300, 8
	idx := buildTestIndex(n, dim)

	path := filepath.Join(t.TempDir(), "index.vdb")
	if err := idx.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading file back: %v", err)
	}
	if !bytes.HasPrefix(data, fileMagic[:]) {
		t.Fatalf("file does not start with the magic bytes")
	}

	var graph bytes.Buffer
	if err := writeGraph(&graph, idx.neighbors); err != nil {
		t.Fatalf("writeGraph: %v", err)
	}
	want := 32 + 4*n*dim + graph.Len() // header + vectors + graph
	if len(data) != want {
		t.Fatalf("file is %d bytes, want %d (a short file usually means a missing Flush)", len(data), want)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	const n, dim = 500, 8
	idx := buildTestIndex(n, dim)
	path := filepath.Join(t.TempDir(), "index.vdb")
	if err := idx.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path, distance.L2Squared)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Same settings, vectors and links.
	if loaded.m != idx.m || loaded.efConstruction != idx.efConstruction ||
		loaded.entry != idx.entry || loaded.maxLevel != idx.maxLevel {
		t.Fatalf("settings differ: got m=%d efc=%d entry=%d maxLevel=%d, want m=%d efc=%d entry=%d maxLevel=%d",
			loaded.m, loaded.efConstruction, loaded.entry, loaded.maxLevel,
			idx.m, idx.efConstruction, idx.entry, idx.maxLevel)
	}
	if loaded.store.Dim() != dim || !slices.Equal(loaded.store.Data(), idx.store.Data()) {
		t.Fatal("vectors differ after load")
	}
	if !sameGraph(loaded.neighbors, idx.neighbors) {
		t.Fatal("links differ after load")
	}

	// Same answers to the same queries.
	rng := rand.New(rand.NewSource(7))
	for range 50 {
		q := make([]float32, dim)
		for j := range q {
			q[j] = rng.Float32()
		}
		want := idx.Search(q, 10, 50)
		got := loaded.Search(q, 10, 50)
		if !slices.Equal(got, want) {
			t.Fatalf("search results differ:\n got %v\nwant %v", got, want)
		}
	}

	// Insert still works, and does not touch the original index.
	v := make([]float32, dim)
	for j := range v {
		v[j] = 0.5
	}
	if id := loaded.Insert(v); id != n {
		t.Fatalf("Insert after Load returned %d, want %d", id, n)
	}
	if loaded.store.Len() != n+1 || len(loaded.neighbors) != n+1 {
		t.Fatalf("after Insert: %d vectors, %d nodes, want %d of each", loaded.store.Len(), len(loaded.neighbors), n+1)
	}
	if len(loaded.neighbors[n][0]) == 0 {
		t.Fatal("the inserted node has no links")
	}
	if got := loaded.Search(v, 10, 50); len(got) != 10 {
		t.Fatalf("Search after Insert returned %d results, want 10", len(got))
	}
	if idx.store.Len() != n {
		t.Fatalf("original index changed: %d vectors, want %d", idx.store.Len(), n)
	}
}

func TestLoadRejectsBadFiles(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.vdb")
	if err := buildTestIndex(100, 8).Save(good); err != nil {
		t.Fatalf("Save: %v", err)
	}
	data, err := os.ReadFile(good)
	if err != nil {
		t.Fatalf("reading file back: %v", err)
	}

	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatalf("writing %s: %v", name, err)
		}
		return p
	}

	badEntry := slices.Clone(data)
	binary.LittleEndian.PutUint32(badEntry[24:], 9999) // entry field

	cases := []struct {
		name string
		path string
	}{
		{"missing file", filepath.Join(dir, "nope.vdb")},
		{"empty file", write("empty.vdb", nil)},
		{"not an index file", write("text.vdb", []byte("this is just some text, long enough to fill a header"))},
		{"cut inside the vectors", write("cut1.vdb", data[:100])},
		{"cut inside the graph", write("cut2.vdb", data[:len(data)-5])},
		{"entry past the last node", write("entry.vdb", badEntry)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			idx, err := Load(c.path, distance.L2Squared)
			if err == nil {
				t.Fatalf("got an index with %d nodes, want an error", len(idx.neighbors))
			}
			t.Logf("error: %v", err)
		})
	}
}

func TestSaveReplacesFileAtomically(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.vdb")

	if err := buildTestIndex(100, 8).Save(path); err != nil {
		t.Fatalf("first Save: %v", err)
	}
	oldBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// A reader that opened the file before the second Save.
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	if err := buildTestIndex(200, 8).Save(path); err != nil {
		t.Fatalf("second Save: %v", err)
	}

	// The old reader still sees the complete old file.
	got, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, oldBytes) {
		t.Fatal("the old file was modified in place; Save must replace it by rename")
	}

	// The name now points at the new index.
	loaded, err := Load(path, distance.L2Squared)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.store.Len() != 200 {
		t.Fatalf("loaded %d vectors, want 200", loaded.store.Len())
	}

	// No temporary file is left behind.
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("temporary file still exists (stat error: %v)", err)
	}
}

func TestSaveCleansUpWhenRenameFails(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "taken")
	if err := os.Mkdir(target, 0o755); err != nil { // a directory is in the way
		t.Fatal(err)
	}
	err := buildTestIndex(50, 8).Save(target)
	if err == nil {
		t.Fatal("want an error when the target is a directory")
	}
	t.Logf("error: %v", err)
	if _, statErr := os.Stat(target + ".tmp"); !os.IsNotExist(statErr) {
		t.Fatalf("temporary file left behind (stat error: %v)", statErr)
	}
}
