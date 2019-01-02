package jnet

import "math"

type Neuron struct {
	*Logger
	Connections []Connection
	Value       float64
	Result      string
	Transform   NeuralTransformer
}

type NeuralConstructor func() (nn *Neuron)

func newNeuron(nt NeuralTransformer) (nn *Neuron) {
	return &Neuron{
		Logger:    NewLogger("Neuron", DefaultPadding),
		Transform: nt,
	}
}

func NewNilNeuron() (nn *Neuron) {
	return newNeuron(nil)
}

func NewSigmoidNeuron() (nn *Neuron) {
	return newNeuron(sigmoid)
}

func NewRectifiedLinearUnitNeuron() (nn *Neuron) {
	return newNeuron(rectifiedLinearUnit)
}

func NewPositiveBinaryNeuron() (nn *Neuron) {
	return newNeuron(positiveBinary)
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
