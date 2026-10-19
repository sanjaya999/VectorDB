package distance

import (
	"log"
	"math"
)

//Dot product implementation

func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		log.Panic("vectors must be the same length")
	}
	var sum float32
	for i := range len(a) {
		sum += a[i] * b[i]
	}
	return sum
}

// L2Squared returns the squared Euclidean distance between a and b.
// Square root is not done to avoid computational overhead.
// a > b then sqrt of a > sqrt of b so, we can compare the squared values directly.
func L2Squared(a, b []float32) float32 {
	if len(a) != len(b) {
		log.Panic("vectors must be the same length")
	}
	var sum float32
	for i := range len(a) {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// cosine similarity implementation
// Cosine returns the cosine similarity of a and b, or 0 if either is a
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		log.Panic("vectors must be the same length")
	}
	dotProduct := Dot(a, b)
	var magnitudeA float32
	var magnitudeB float32
	for i := range len(a) {
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}
	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}
	return dotProduct / float32(math.Pow(float64(magnitudeA), 0.5)*math.Pow(float64(magnitudeB), 0.5))
}
