package main

import (
	"fmt"
	"jnet/pkg/data"
	"jnet/pkg/layer"
	"jnet/pkg/network"
	"jnet/pkg/neuron"
)

func main() {
	nw := network.New([]layer.Layer{
		*layer.New(1, neuron.TypeInput),
		*layer.New(1, neuron.TypeTwoTimes),
		*layer.New(1, neuron.TypeExp),
		*layer.New(1, neuron.TypeOnePlus),
		*layer.New(1, neuron.TypeInverse),
		*layer.New(1, neuron.TypeNegate),
		*layer.New(1, neuron.TypeOutput).
			SetOutputNeuronResults(
				[]string{"output???"},
			),
	}).RandomizeConnectionWeights()

	nw.Train([]data.TrainingData{
		{
			Truth: data.T{-0.88079708},
			Data:  data.D{-1.0},
		},
		{
			Truth: data.T{-0.73105858},
			Data:  data.D{-0.5},
		},
		{
			Truth: data.T{-0.59868766},
			Data:  data.D{-0.2},
		},
		{
			Truth: data.T{-0.549834},
			Data:  data.D{-0.1},
		},
		{
			Truth: data.T{-0.5},
			Data:  data.D{0.0},
		},
		{
			Truth: data.T{-0.450166},
			Data:  data.D{0.1},
		},
		{
			Truth: data.T{-0.40131234},
			Data:  data.D{0.2},
		},
		{
			Truth: data.T{-0.26894142},
			Data:  data.D{0.5},
		},
		{
			Truth: data.T{-0.11920292},
			Data:  data.D{1.0},
		},
	})

	nw.Predict(data.D{0.0})

	fmt.Println(nw.GetResults())
}
