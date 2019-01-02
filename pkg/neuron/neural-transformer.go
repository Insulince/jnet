package neuron

import (
	"jnet/pkg/util"
	"math"
)

type NeuralTransformer func(x float64) (y float64)

func sigmoid(x float64) (y float64) {
	return (util.Max-util.Min)/(1+math.Pow(math.E, -x)) + util.Min
}

func rectifiedLinearUnit(x float64) (y float64) {
	if x > util.Cutoff {
		return x
	}
	return util.Cutoff
}

func positiveBinary(x float64) (y float64) {
	if x > util.Cutoff {
		return util.Max
	}
	return util.Cutoff
}
