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
	TypeTwoTimes            = "twoTimes"
	TypeExp                 = "exp"
	TypeOnePlus             = "onePlus"
	TypeInverse             = "inverse"
	TypeNegate              = "negate"
	TypeOutput              = "output"
)

type Neuron struct {
	*util.Logger
	Type            string
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
	case TypeTwoTimes:
		nt = twoTimes
		nat = dTwoTimes
	case TypeExp:
		nt = exp
		nat = dExp
	case TypeOnePlus:
		nt = onePlus
		nat = dOnePlus
	case TypeInverse:
		nt = inverse
		nat = dInverse
	case TypeNegate:
		nt = negate
		nat = dNegate
	case TypeOutput:
		nt = passThrough
		nat = dPassThrough
	default:
		panic("Unrecognized neuron type provided")
	}

	return &Neuron{
		Logger:       util.NewLogger("Neuron", util.DefaultPadding),
		Type:         t,
		Transform:    nt,
		AntiTransorm: nat,
	}
}

func (n *Neuron) StageForPass() {
	if n.Type != TypeInput {
		n.Value = 0.0
	}
	n.Sum = 0.0
	n.LossGradients = []data.Gradient{}
	n.LossGradientSum = 0.0
	n.LocalGradients = []data.Gradient{}
}
