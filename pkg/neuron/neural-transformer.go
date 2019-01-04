package neuron

import (
	"math"
)

type NeuralTransformer func(x float64) (y float64)

func sigmoid(x float64) (y float64) {
	// Old, more generic version
	//return (util.Max-util.Min)/(1+math.Pow(math.E, -x)) + util.Min

	// Specifically bounded version.
	return 2/(1+math.Pow(math.E, -x)) - 1
}

func rectifiedLinearUnit(x float64) (y float64) {
	if x > 0 {
		return x
	}
	return 0
}

func passThrough(x float64) (y float64) {
	return x
}

/////////////// ANTI TRANSFORMS ///////////////

//// For the original vanilla sigmoid function, not our custom bounds one.
//func dSigmoid(x float64) (y float64) {
//	return sigmoid(x) * (1 - sigmoid(x))
//}

func dSigmoid(x float64) (y float64) {
	return 2 * math.Pow(math.E, x) / math.Pow(math.Pow(math.E, x)+1, 2)
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
