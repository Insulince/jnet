package jnet

import (
	"errors"
	"math/rand"
)

type TrainingConfiguration struct {
	LearningRate       float64 `json:"learningRate"`
	TrainingIterations int     `json:"trainingIterations"`
	MiniBatchSize      int     `json:"miniBatchSize"`
	AverageLossCutoff  float64 `json:"averageLossCutoff"`
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

type HumanDatum struct {
	Data  [][]float64
	Truth []float64
}

func (hd *HumanDatum) ToTrainingDatum() (td *TrainingDatum) {
	td = &TrainingDatum{}

	qdr := len(hd.Data)
	for dri := 0; dri < qdr; dri++ { // For every row of data in this human datum...
		dr := hd.Data[dri]

		qd := len(dr)
		for di := 0; di < qd; di++ { // For every cell of data in this row...
			dataCell := dr[di]

			td.Data = append(td.Data, dataCell)
		}
	}

	td.Truth = hd.Truth

	return td
}

type HumanData []HumanDatum

func (hd *HumanData) ToTrainingData() (td TrainingData) {
	td = []TrainingDatum{}

	qhd := len(*hd)
	for hdi := 0; hdi < qhd; hdi++ { // For every human datum in this human data...
		hdatum := (*hd)[hdi]

		td = append(td, *hdatum.ToTrainingDatum())
	}

	return td
}
