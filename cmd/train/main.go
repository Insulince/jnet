package main

import (
	"fmt"
	"jnet/pkg/data"
	"jnet/pkg/layer"
	"jnet/pkg/network"
	"jnet/pkg/neuron"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func main() {
	nw := network.New([]layer.Layer{
		*layer.New(4, neuron.TypeInput),
		*layer.New(4, neuron.TypeSigmoid),
		*layer.New(4, neuron.TypeSigmoid),
		*layer.New(8, neuron.TypeRectifiedLinearUnit),
		*layer.New(4, neuron.TypeOutput).
			SetOutputNeuronResults(
				[]string{"Solid", "Vertical", "Diagonal", "Horizontal"},
			),
	}).RandomizeConnectionWeights()

	nw.Train([]data.TrainingData{
		{
			Truth: data.T{1.0, 0.0, 0.0, 0.0}, // Solid - White
			Data:  data.D{1.0, 1.0, 1.0, 1.0},
		},
		{
			Truth: data.T{1.0, 0.0, 0.0, 0.0}, // Solid - Black
			Data:  data.D{-1.0, -1.0, -1.0, -1.0},
		},
		{
			Truth: data.T{0.0, 1.0, 0.0, 0.0}, // Vertical - White Left
			Data:  data.D{1.0, -1.0, -1.0, 1.0},
		},
		{
			Truth: data.T{0.0, 1.0, 0.0, 0.0}, // Vertical - Black Left
			Data:  data.D{-1.0, 1.0, 1.0, -1.0},
		},
		{
			Truth: data.T{0.0, 0.0, 1.0, 0.0}, // Diagonal - White Forward
			Data:  data.D{1.0, -1.0, 1.0, -1.0},
		},
		{
			Truth: data.T{0.0, 0.0, 1.0, 0.0}, // Diagonal - Black Forward
			Data:  data.D{-1.0, 1.0, -1.0, 1.0},
		},
		{
			Truth: data.T{0.0, 0.0, 0.0, 1.0}, // Horizontal - White Top
			Data:  data.D{1.0, 1.0, -1.0, -1.0},
		},
		{
			Truth: data.T{0.0, 0.0, 0.0, 1.0}, // Horizontal - Black Top
			Data:  data.D{-1.0, -1.0, 1.0, 1.0},
		},
	})

	nw.Predict(data.D{1.0, -1.0, 1.0, -1.0}) // Diagonal - White Forward

	fmt.Println(nw.GetResults())
}
