package jnet

import "math"

func sigmoid(x float64) (y float64) {
	return 1 / (1 + math.Pow(math.E, -x))
}

func tanh(x float64) (y float64) {
	return math.Tanh(x)
}

func relu(x float64) (y float64) {
	if x > 0 {
		return x
	}
	return 0
}

func linear(x float64) (y float64) {
	return x
}
