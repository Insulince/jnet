package neuron

import (
	"jnet/pkg/connection"
	"jnet/pkg/util"
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
