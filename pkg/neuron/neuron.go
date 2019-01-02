package neuron

import (
	"jnet/pkg/connection"
	"jnet/pkg/util"
	"math"
)

const (
	TypeNil                 = "nil"
	TypeSigmoid             = "sigmoid"
	TypeRectifiedLinearUnit = "rectifiedLinearUnit"
	TypePositiveBinary      = "positiveBinary"
)

type Neuron struct {
	*util.Logger
	Connections []connection.Connection
	Value       float64
	Result      string
	Transform   NeuralTransformer
}

func New(t string) (nn *Neuron) {
	var nt NeuralTransformer

	switch t {
	case TypeNil:
		nt = nil
	case TypeSigmoid:
		nt = sigmoid
	case TypeRectifiedLinearUnit:
		nt = rectifiedLinearUnit
	case TypePositiveBinary:
		nt = positiveBinary
	default:
		panic("Unrecognized neuron type provided")
	}

	return &Neuron{
		Logger:    util.NewLogger("Neuron", util.DefaultPadding),
		Transform: nt,
	}
}

type NeuralTransformer func(x float64) (y float64)

func sigmoid(x float64) (y float64) {
	return 2/(1+math.Pow(math.E, -x)) - 1
}

func rectifiedLinearUnit(x float64) (y float64) {
	if x > 0 {
		return x
	}
	return 0
}

func positiveBinary(x float64) (y float64) {
	if x > 0 {
		return 1
	}
	return 0
}
