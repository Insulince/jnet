package jnet

import (
	"errors"
	"math/rand"
)

type TrainingConfiguration struct {
	LearningRate      float64 `json:"learningRate"`
	Iterations        int     `json:"iterations"`
	MiniBatchSize     int     `json:"miniBatchSize"`
	AverageLossCutoff float64 `json:"averageLossCutoff"`
}

type TrainingDatum struct {
	Data  []float64 `json:"data"`
	Truth []float64 `json:"truth"`
}

type TrainingData []TrainingDatum

func (td *TrainingData) shuffle() {
	qtd := len(*td)
	for tdi := 0; tdi < qtd; tdi++ { // For every training datum in this training data...
		rtdi := rand.Intn(tdi + 1) // Select a random training datum index in [0, tdi]

		(*td)[tdi], (*td)[rtdi] = (*td)[rtdi], (*td)[tdi]
	}
}

func (td *TrainingData) miniBatch(miniBatchSize int) (mb TrainingData, err error) {
	if miniBatchSize > len(*td) {
		return nil, errors.New("requested mini batch size larger than number of training datums")
	}

	td.shuffle()

	return (*td)[0:miniBatchSize], nil
}
