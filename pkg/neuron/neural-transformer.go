package neuron

import (
	"jnet/pkg/data"
	"math"
)

type NeuralTransformer func(x data.V) (y data.V)

func sigmoid(x data.V) (y data.V) {
	// Old, more generic version
	//return (util.Max-util.Min)/(1+math.Pow(math.E, -x)) + util.Min

	// Specifically bounded version.
	return data.V(2/(1+math.Pow(math.E, float64(-x))) - 1)
}

func rectifiedLinearUnit(x data.V) (y data.V) {
	if x > 0 {
		return x
	}
	return 0
}

func passThrough(x data.V) (y data.V) {
	return x
}

/////////////// ANTI TRANSFORMS ///////////////

//// For the original vanilla sigmoid function, not our custom bounds one.
//func dSigmoid(x data.V) (y data.V) {
//	return sigmoid(x) * (1 - sigmoid(x))
//}

func dSigmoid(x data.V) (y data.V) {
	return data.V(2 * math.Pow(math.E, float64(x)) / math.Pow(math.Pow(math.E, float64(x))+1, 2))
}

func dRectifiedLinearUnit(x data.V) (y data.V) {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

func dPassThrough(x data.V) (y data.V) {
	return 1.0
}
