package store

import "log"

type Store struct {
	//dimensions of the vector
	dim int
	//data is the actual vector data
	data []float32
}

type Result struct {
	//ID is the ID of the vector
	ID int
	//Dist is the distance of the vector from the query vector
	Dist float32
}

// New creates a new Store with the given dimension
func New(dim int) *Store {
	if dim <= 0 {
		log.Panic("dimenstion should be positive")
	}
	return &Store{
		dim: dim,
		//In GO uninitialized slices are nil
	}
}

// Len returns the number of vectors in the store
// Since our store is a flat array , we dont take len ,
// we divide len by the dimenstion to get total number of vectors
func (s *Store) Len() int {
	return len(s.data) / s.dim
}

// Add adds a new vector to the store and returns its ID
// Len is taken because it helps to differentiate between the vectors
func (s *Store) Add(v []float32) int {
	if len(v) != s.dim {
		log.Panic("vector dimension does not match store dimension")
	}
	s.data = append(s.data, v...)
	length := s.Len()
	return length - 1
}

// Get returns the vector with the given ID
// It returens a slice of float32 which is the vector data
func (s *Store) Get(id int) []float32 {
	if id < 0 || id >= s.Len() {
		log.Panic("invalid vector ID")
	}
	start := id * s.dim
	end := start + s.dim
	return s.data[start:end]
}
