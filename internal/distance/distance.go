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
	b = b[:len(a)]

	var s0, s1, s2, s3 float32 // four separate running totals
	i := 0
	for ; i+4 <= len(a); i += 4 { // take four numbers per step
		a4 := a[i : i+4 : i+4]
		b4 := b[i : i+4 : i+4]
		d0 := a4[0] - b4[0]
		d1 := a4[1] - b4[1]
		d2 := a4[2] - b4[2]
		d3 := a4[3] - b4[3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ { // leftovers, when the length isn't a multiple of 4
		d := a[i] - b[i]
		s0 += d * d
	}
	return s0 + s1 + s2 + s3
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

// CosineDistance turns similarity into a distance: lower = more similar.
// Same direction -> 0, perpendicular -> 1, opposite -> 2.
func CosineDistance(a, b []float32) float32 {
	return 1 - CosineSimilarity(a, b)
}
