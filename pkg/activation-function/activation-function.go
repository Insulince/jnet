package activationfunction

import "math"

type ActivationFunction func(x float64) float64

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func Relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func Linear(x float64) float64 {
	return x
}

// NOTE(justin): The following ensures that all functions adhere to the ActivationFunction type
var (
	_ ActivationFunction = Sigmoid
	_ ActivationFunction = Tanh
	_ ActivationFunction = Relu
	_ ActivationFunction = Linear
)
