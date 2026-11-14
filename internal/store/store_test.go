package store

import (
	"slices"
	"testing"
)

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic", name)
		}
	}()
	f()
}

func TestAddAndGet(t *testing.T) {
	s := New(3)
	id0 := s.Add([]float32{1, 2, 3})
	id1 := s.Add([]float32{4, 5, 6})

	if id0 != 0 || id1 != 1 {
		t.Fatalf("ids = %d, %d; want 0, 1", id0, id1)
	}
	got := s.Get(1)
	want := []float32{4, 5, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Get(1)[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestLen(t *testing.T) {
	s := New(2)
	if s.Len() != 0 {
		t.Errorf("new store Len = %d, want 0", s.Len())
	}
	s.Add([]float32{1, 2})
	s.Add([]float32{3, 4})
	if s.Len() != 2 {
		t.Errorf("Len = %d, want 2", s.Len())
	}
}

func TestPanics(t *testing.T) {
	mustPanic(t, "New(0)", func() { New(0) })
	mustPanic(t, "Add wrong dim", func() { New(3).Add([]float32{1, 2}) })

	s := New(2)
	s.Add([]float32{1, 2})
	mustPanic(t, "Get(-1)", func() { s.Get(-1) })
	mustPanic(t, "Get(Len)", func() { s.Get(s.Len()) })
}

// Appending to a vector returned by Get must not overwrite its neighbor.
func TestGetCannotCorruptNeighbor(t *testing.T) {
	s := New(2)
	s.Add([]float32{1, 2})
	s.Add([]float32{3, 4})

	_ = append(s.Get(0), 99)

	got := s.Get(1)
	if got[0] != 3 || got[1] != 4 {
		t.Errorf("vector 1 corrupted: %v", got)
	}
}

func TestDimAndData(t *testing.T) {
	s := New(3)
	s.Add([]float32{1, 2, 3})
	s.Add([]float32{4, 5, 6})

	if s.Dim() != 3 {
		t.Fatalf("Dim() = %d, want 3", s.Dim())
	}
	got := s.Data()
	if want := []float32{1, 2, 3, 4, 5, 6}; !slices.Equal(got, want) {
		t.Fatalf("Data() = %v, want %v", got, want)
	}
	if cap(got) != len(got) {
		t.Errorf("Data() len %d, cap %d: capacity must be capped", len(got), cap(got))
	}
}

func TestFromDataRoundTrip(t *testing.T) {
	const dim, n = 4, 50
	orig := New(dim)
	for i := range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i*dim + j)
		}
		orig.Add(v)
	}

	loaded, err := FromData(orig.Dim(), orig.Data())
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	if loaded.Dim() != dim || loaded.Len() != n {
		t.Fatalf("got dim %d len %d, want %d and %d", loaded.Dim(), loaded.Len(), dim, n)
	}
	for id := range n {
		if !slices.Equal(loaded.Get(id), orig.Get(id)) {
			t.Fatalf("vector %d: got %v, want %v", id, loaded.Get(id), orig.Get(id))
		}
	}

	// Add continues the ID sequence and leaves the original alone.
	v := []float32{-1, -2, -3, -4}
	if id := loaded.Add(v); id != n {
		t.Errorf("Add after FromData returned %d, want %d", id, n)
	}
	if !slices.Equal(loaded.Get(n), v) {
		t.Errorf("Get(%d) = %v, want %v", n, loaded.Get(n), v)
	}
	if orig.Len() != n {
		t.Errorf("original Len changed to %d", orig.Len())
	}
}

func TestFromDataDoesNotUseCallerCapacity(t *testing.T) {
	buf := make([]float32, 6, 12) // two 3-d vectors, room for two more
	copy(buf, []float32{1, 2, 3, 4, 5, 6})
	s, err := FromData(3, buf)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	s.Add([]float32{7, 8, 9})
	buf = append(buf, 99) // lands in buf's spare capacity
	_ = buf
	if got := s.Get(2); !slices.Equal(got, []float32{7, 8, 9}) {
		t.Errorf("Get(2) = %v, want [7 8 9]: store wrote into the caller's spare capacity", got)
	}
}

func TestFromDataEmpty(t *testing.T) {
	s, err := FromData(8, nil)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	if s.Len() != 0 || s.Dim() != 8 {
		t.Fatalf("got len %d dim %d, want 0 and 8", s.Len(), s.Dim())
	}
	if id := s.Add(make([]float32, 8)); id != 0 {
		t.Errorf("first Add returned %d, want 0", id)
	}
}

func TestFromDataRejectsBadInput(t *testing.T) {
	cases := []struct {
		name string
		dim  int
		data []float32
	}{
		{"zero dim", 0, []float32{1, 2}},
		{"negative dim", -3, []float32{1, 2, 3}},
		{"length not a multiple of dim", 4, make([]float32, 10)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if s, err := FromData(c.dim, c.data); err == nil {
				t.Errorf("got store %+v, want an error", s)
			}
		})
	}
}
