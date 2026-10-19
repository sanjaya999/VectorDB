package distance

import (
	"math"
	"testing"
)

func approx(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-6
}

func TestDot(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"basic", []float32{1, 2, 3}, []float32{4, 5, 6}, 32},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"negative", []float32{1, -2}, []float32{3, 4}, -5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Dot(tt.a, tt.b); !approx(got, tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"perpendicular", []float32{1, 0, 0}, []float32{0, 1, 0}, 0},
		{"same direction", []float32{1, 2}, []float32{2, 4}, 1},
		{"opposite", []float32{1, 2}, []float32{-1, -2}, -1},
		{"zero vector", []float32{0, 0}, []float32{1, 2}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CosineSimilarity(tt.a, tt.b); !approx(got, tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMismatchedLengthPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on mismatched lengths")
		}
	}()
	Dot([]float32{1, 2}, []float32{1})
}
