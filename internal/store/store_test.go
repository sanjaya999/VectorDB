package store

import "testing"

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
