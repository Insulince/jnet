package train

import (
	"errors"
	"math/rand"
)

type Configuration struct {
	LearningRate      float64
	Iterations        int
	MiniBatchSize     int
	AverageLossCutoff float64
}

type Datum struct {
	Data  []float64
	Truth []float64
}

type Data []Datum

func (d Data) shuffle() {
	for i := 0; i < len(d); i++ { // For every training datum in this training data...
		r := rand.Intn(i + 1) // Select a random training datum index in [0, i]

		d[i], d[r] = d[r], d[i]
	}
}

func (d Data) MiniBatch(miniBatchSize int) (Data, error) {
	if miniBatchSize < 1 {
		return nil, errors.New("requested mini batch size must be at least 1")
	}
	if miniBatchSize > len(d) {
		return nil, errors.New("requested mini batch size larger than number of training datums")
	}

	d.shuffle()

	return d[:miniBatchSize], nil
}
