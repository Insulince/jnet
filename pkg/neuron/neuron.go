package neuron

import (
	"jnet/pkg/connection"
	"jnet/pkg/util"
)

const (
	TypeInput               = "input"
	TypeSigmoid             = "sigmoid"
	TypeRectifiedLinearUnit = "rectifiedLinearUnit"
	TypeOutput              = "output"
)

type Neuron struct {
	*util.Logger
	Connections    []connection.Connection
	Value          float64
	Sum            float64
	Result         string
	Transform      NeuralTransformer
	AntiTransorm   NeuralTransformer
	LossGradient   float64   // d(loss)/d(me)
	LocalGradients []float64 // d(me)/d(n_i-1_k), d(me)/d(n_i-1_k+1), d(me)/d(n_i-1_k+2), ...
}

func New(t string) (nn *Neuron) {
	var nt NeuralTransformer
	var nat NeuralTransformer

	switch t {
	case TypeInput:
		nt = nil
		nat = nil
	case TypeSigmoid:
		nt = sigmoid
		nat = dSigmoid
	case TypeRectifiedLinearUnit:
		nt = rectifiedLinearUnit
		nat = dRectifiedLinearUnit
	case TypeOutput:
		nt = passThrough
		nat = dPassThrough
	default:
		panic("Unrecognized neuron type provided")
	}

	return &Neuron{
		Logger:       util.NewLogger("Neuron", util.DefaultPadding),
		Transform:    nt,
		AntiTransorm: nat,
	}
}
