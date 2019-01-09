package neuron

import (
	"jnet/pkg/data"
	"math"
)

type NeuralTransformer func(x data.V) (y data.V)

func sigmoid(x data.V) (y data.V) {
	return data.V(2/(1+math.Pow(math.E, float64(-x))) - 1)
}

func dSigmoid(x data.V) (y data.V) {
	return data.V(2 * math.Pow(math.E, float64(x)) / math.Pow(math.Pow(math.E, float64(x))+1, 2))
}

func rectifiedLinearUnit(x data.V) (y data.V) {
	if x > 0 {
		return x
	}
	return 0
}

func dRectifiedLinearUnit(x data.V) (y data.V) {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

func passThrough(x data.V) (y data.V) {
	return x
}

func dPassThrough(x data.V) (y data.V) {
	return 1.0
}

func twoTimes(x data.V) (y data.V) {
	return 2 * x
}

func dTwoTimes(x data.V) (y data.V) {
	return 2
}

func exp(x data.V) (y data.V) {
	return data.V(math.Pow(math.E, float64(x)))
}

func dExp(x data.V) (y data.V) {
	return data.V(math.Pow(math.E, float64(x)))
}

func onePlus(x data.V) (y data.V) {
	return 1 + x
}

func dOnePlus(x data.V) (y data.V) {
	return 1
}

func inverse(x data.V) (y data.V) {
	return 1.0 / x
}

func dInverse(x data.V) (y data.V) {
	return data.V(-1.0 / math.Pow(float64(x), -2))
}

func negate(x data.V) (y data.V) {
	return -1 * x
}

func dNegate(x data.V) (y data.V) {
	return -1
}
