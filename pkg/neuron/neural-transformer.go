package neuron

import (
	"math"
)

const (
	Max    = 1.0
	Min    = -1.0
	Cutoff = (Max-Min)/2 + Min
)

type NeuralTransformer func(x float64) (y float64)

func sigmoid(x float64) (y float64) {
	return (Max-Min)/(1+math.Pow(math.E, -x)) + Min
}

func rectifiedLinearUnit(x float64) (y float64) {
	if x > Cutoff {
		return x
	}
	return Cutoff
}

func positiveBinary(x float64) (y float64) {
	if x > Cutoff {
		return Max
	}
	return Cutoff
}
