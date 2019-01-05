package neuron

import (
	"jnet/pkg/connection"
	"jnet/pkg/data"
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
	Connections     []connection.Connection
	Value           data.V
	Sum             data.V
	Result          string
	Transform       NeuralTransformer
	AntiTransorm    NeuralTransformer
	LossGradients   []data.Gradient
	LossGradientSum data.Gradient
	LocalGradients  []data.Gradient
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

func (n *Neuron) StageForPass() {
	n.Value = 0.0
	n.Sum = 0.0
	n.LossGradients = []data.Gradient{}
	n.LossGradientSum = 0.0
	n.LocalGradients = []data.Gradient{}
}
