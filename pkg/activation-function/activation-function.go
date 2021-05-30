// Package activationfunction is for isolating the concept of an activation
// function into its own area. Activation functions are used in neural networks
// to provide stability to their inputs and outputs through non-linearization.
// What that means is that a neuron which is loaded up with tons of values in
// its weighted sum will be sent through an activation function which sets a
// bound on it to prevent this large value from influencing the rest of the
// network.
package activationfunction

import (
	"fmt"
	"math"

	"github.com/pkg/errors"
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

// MustGetFunction calls GetFunction but panics if an error is encountered.
func MustGetFunction(name Name) ActivationFunction {
	fn, err := GetFunction(name)
	if err != nil {
		panic(errors.Wrap(err, "must get function"))
	}
	return fn
}

func ErrNotFound(name Name) error {
	return fmt.Errorf("no activation function found with name \"%v\"", name)
}

// Range: [0, 0]
func noop(x float64) float64 {
	return 0
}

// Range: (1, 1)
func sigmoid(x float64) float64 {
	return 2/(1+math.Pow(math.E, -x)) - 1
}

// Range: (-1, 1)
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// Range: [0, +inf) UNBOUNDED
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Range: (-inf, +inf) UNBOUNDED
func linear(x float64) float64 {
	return x
}

// NOTE(justin): The following ensures that all functions adhere to the
// ActivationFunction type
var (
	_ ActivationFunction = noop
	_ ActivationFunction = sigmoid
	_ ActivationFunction = tanh
	_ ActivationFunction = relu
	_ ActivationFunction = linear
)
