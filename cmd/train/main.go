package main

import (
	"fmt"
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

	nw.Train([]network.TD{
		{
			Truth: []float64{1.0, 0.0, 0.0, 0.0}, // Solid - White
			Data:  []float64{1, 1, 1, 1},
		},
		{
			Truth: []float64{1.0, 0.0, 0.0, 0.0}, // Solid - Black
			Data:  []float64{-1, -1, -1, -1},
		},
		{
			Truth: []float64{0.0, 1.0, 0.0, 0.0}, // Vertical - White Left
			Data:  []float64{1, -1, -1, 1},
		},
		{
			Truth: []float64{0.0, 1.0, 0.0, 0.0}, // Vertical - Black Left
			Data:  []float64{-1, 1, 1, -1},
		},
		{
			Truth: []float64{0.0, 0.0, 1.0, 0.0}, // Diagonal - White Forward
			Data:  []float64{1, -1, 1, -1},
		},
		{
			Truth: []float64{0.0, 0.0, 1.0, 0.0}, // Diagonal - Black Forward
			Data:  []float64{-1, 1, -1, 1},
		},
		{
			Truth: []float64{0.0, 0.0, 0.0, 1.0}, // Horizontal - White Top
			Data:  []float64{1, 1, -1, -1},
		},
		{
			Truth: []float64{0.0, 0.0, 0.0, 1.0}, // Horizontal - Black Top
			Data:  []float64{-1, -1, 1, 1},
		},
	})

	nw.Predict([]float64{1, -1, 1, -1}) // Diagonal - White Forward

	fmt.Println(nw.GetResults())
}
