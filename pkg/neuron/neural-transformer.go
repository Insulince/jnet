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
	if x > 0 {
		return x
	}
	return util.Midpoint
}

func passThrough(x float64) (y float64) {
	return x
}

//////////

func dSigmoid(x float64) (y float64) {
	return sigmoid(x) * (1 - sigmoid(x))
}

func dRectifiedLinearUnit(x float64) (y float64) {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

func dPassThrough(x float64) (y float64) {
	return 1.0
}
