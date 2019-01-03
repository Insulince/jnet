package main

import (
	"fmt"
	"jnet/pkg/layer"
	"jnet/pkg/network"
	"jnet/pkg/neuron"
)

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

	td := []network.TD{
		{
			Truth: []float64{1.0, 0.0, 0.0, 0.0},
			Data:  []float64{1, 1, 1, 1},
		},
		{
			Truth: []float64{1.0, 0.0, 0.0, 0.0},
			Data:  []float64{-1, -1, -1, -1},
		},
		{
			Truth: []float64{0.0, 1.0, 0.0, 0.0},
			Data:  []float64{-1, 1, 1, -1},
		},
		{
			Truth: []float64{0.0, 1.0, 0.0, 0.0},
			Data:  []float64{1, -1, -1, 1},
		},
		{
			Truth: []float64{0.0, 0.0, 1.0, 0.0},
			Data:  []float64{-1, 1, -1, 1},
		},
		{
			Truth: []float64{0.0, 0.0, 1.0, 0.0},
			Data:  []float64{1, -1, 1, -1},
		},
		{
			Truth: []float64{0.0, 0.0, 0.0, 1.0},
			Data:  []float64{1, 1, -1, -1},
		},
		{
			Truth: []float64{0.0, 0.0, 0.0, 1.0},
			Data:  []float64{-1, -1, 1, 1},
		},
	}

	nw.Train(td)

	nw.Process([]float64{1, -1, 1, -1})

	fmt.Println(nw.Results())
}
