package jnet

import "math/rand"

type TrainingConfiguration struct {
	LearningRate       float64
	TrainingIterations int
	MiniBatchSize      int
	AverageLossCutoff  float64
}

type TrainingDatum struct {
	Data  []float64
	Truth []float64
}

type TrainingData []TrainingDatum

func (td *TrainingData) shuffle() {
	qtd := len(*td)
	for i := 0; i < qtd; i++ {
		j := rand.Intn(i + 1)

		(*td)[i], (*td)[j] = (*td)[j], (*td)[i]
	}
}

func (td *TrainingData) miniBatch(miniBatchSize int) (mb TrainingData) {
	if miniBatchSize > len(*td) {
		panic("Requested mini batch size larger than number of training datums!")
	}

	td.shuffle()

	return (*td)[0:miniBatchSize]
}

type HumanData struct {
	Data  [][]float64
	Truth []float64
}

func (hd *HumanData) ToTrainingData() (td *TrainingDatum) {
	td = &TrainingDatum{}

	qdr := len(hd.Data)
	for dri := 0; dri < qdr; dri++ {
		dr := hd.Data[dri]

		qd := len(dr)
		for di := 0; di < qd; di++ {
			datum := dr[di]

			td.Data = append(td.Data, datum)
		}
	}

	td.Truth = hd.Truth

	return td
}
