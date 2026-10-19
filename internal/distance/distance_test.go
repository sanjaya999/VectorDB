package distance

import "testing"

func TestDot(t *testing.T) {
	result := Dot(
		[]float32{1, 2, 3},
		[]float32{4, 5, 6},
	)
	expected := float32(32)
	if result != expected {
		t.Errorf("Dot product test failed. Expected %v, got %v", expected, result)
	}
}

func TestL2Squared(t *testing.T) {
	result := L2Squared(
		[]float32{1, 2, 3},
		[]float32{4, 5, 6},
	)
	expected := float32(27)
	if result != expected {
		t.Errorf("L2Squared test failed. Expected %v, got %v", expected, result)
	}
}

func TestCosineSimilarity(t *testing.T) {
	result := CosineSimilarity(
		[]float32{1, 0, 0},
		[]float32{0, 1, 0},
	)
	expected := float32(0)
	if result != expected {
		t.Errorf("CosineSimilarity test failed. Expected %v, got %v", expected, result)
	}
}
