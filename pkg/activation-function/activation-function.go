package activationfunction

import (
	"fmt"
	"math"
)

type (
	Name               string
	ActivationFunction func(x float64) float64
)

const (
	NameNoop    Name = "noop"
	NameSigmoid Name = "sigmoid"
	NameTanh    Name = "tanh"
	NameRelu    Name = "relu"
	NameLinear  Name = "linear"
)

var (
	nameToFunction = map[Name]ActivationFunction{
		NameNoop:    noop,
		NameSigmoid: sigmoid,
		NameTanh:    tanh,
		NameRelu:    relu,
		NameLinear:  linear,
	}
)

func GetFunction(name Name) (ActivationFunction, error) {
	fn, found := nameToFunction[name]
	if !found {
		return nil, ErrNotFound(name)
	}
	return fn, nil
}

func ErrNotFound(name Name) error {
	return fmt.Errorf("no activation function found with name \"%v\"", name)
}

func noop(x float64) float64 {
	return 0
}

func sigmoid(x float64) float64 {
	// NOTE(justin): This is the function for sigmoid from (-1, 1), NOT from (0, 1)
	return 2/(1+math.Pow(math.E, -x)) - 1
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func linear(x float64) float64 {
	return x
}

// NOTE(justin): The following ensures that all functions adhere to the ActivationFunction type
var (
	_ ActivationFunction = noop
	_ ActivationFunction = sigmoid
	_ ActivationFunction = tanh
	_ ActivationFunction = relu
	_ ActivationFunction = linear
)
