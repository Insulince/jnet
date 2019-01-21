package main

import (
	"fmt"
	"jnet"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func main() {
	var humanData = []jnet.HumanData{
		// 0
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
			},
			Truth: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 1, 1, 0},
			},
			Truth: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 1, 1},
			},
			Truth: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		// 1
		{
			Data: [][]float64{
				{1, 0, 0, 0, 0},
				{1, 0, 0, 0, 0},
				{1, 0, 0, 0, 0},
				{1, 0, 0, 0, 0},
				{1, 0, 0, 0, 0},
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 0, 0, 0},
				{0, 1, 0, 0, 0},
				{0, 1, 0, 0, 0},
				{0, 1, 0, 0, 0},
				{0, 1, 0, 0, 0},
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		// 2
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
				{1, 0, 0, 0, 0},
				{1, 1, 1, 0, 0},
			},
			Truth: []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 1, 1, 1, 0},
				{0, 1, 0, 0, 0},
				{0, 1, 1, 1, 0},
			},
			Truth: []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 1, 1},
			},
			Truth: []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		},
		// 3
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
			},
			Truth: []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 1, 1, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 1, 1, 1, 0},
			},
			Truth: []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 1, 1, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 1, 1, 1},
			},
			Truth: []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		},
		// 4
		{
			Data: [][]float64{
				{1, 0, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
			},
			Truth: []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 0, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 1, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
			},
			Truth: []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 0, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 1, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
			},
			Truth: []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		},
		// 5
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{1, 0, 0, 0, 0},
				{1, 1, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 1, 0, 0, 0},
				{0, 1, 1, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 1, 1, 1, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 1, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 1, 1, 1},
			},
			Truth: []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		},
		// 6
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{1, 0, 0, 0, 0},
				{1, 1, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 1, 0, 0, 0},
				{0, 1, 1, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 1, 1, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 1, 1},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		},
		// 7
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 1, 0, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 0, 0, 1, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 0, 0, 1},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		},
		// 8
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 1, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 1, 1, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 1, 1},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		},
		// 9
		{
			Data: [][]float64{
				{1, 1, 1, 0, 0},
				{1, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
				{0, 0, 1, 0, 0},
				{1, 1, 1, 0, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
		},
		{
			Data: [][]float64{
				{0, 1, 1, 1, 0},
				{0, 1, 0, 1, 0},
				{0, 1, 1, 1, 0},
				{0, 0, 0, 1, 0},
				{0, 1, 1, 1, 0},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
		},
		{
			Data: [][]float64{
				{0, 0, 1, 1, 1},
				{0, 0, 1, 0, 1},
				{0, 0, 1, 1, 1},
				{0, 0, 0, 0, 1},
				{0, 0, 1, 1, 1},
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
		},
	}

	var trainingData []jnet.TrainingDatum

	for _, hd := range humanData {
		trainingData = append(trainingData, *hd.ToTrainingData())
	}

	qnils := []int{25, 16, 16, 10}
	labels := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
	nw := jnet.NewNetwork(qnils, labels)

	trainConfig := jnet.TrainingConfiguration{
		LearningRate:       0.1,
		TrainingIterations: 2500000,
		MiniBatchSize:      32,
		AverageLossCutoff:  0.5,
	}
	nw.Train(trainingData, trainConfig)

	hd := jnet.HumanData{
		Data: [][]float64{
			{1, 1, 1, 0, 0},
			{1, 0, 1, 0, 0},
			{1, 1, 1, 0, 0},
			{0, 0, 1, 0, 0},
			{1, 1, 1, 0, 0},
		},
	}
	realData := hd.ToTrainingData().Data

	prediction := nw.Predict(realData)
	fmt.Println(prediction)
	fmt.Println()
}
