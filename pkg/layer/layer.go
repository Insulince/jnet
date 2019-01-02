package layer

import "jnet/pkg/neuron"

type Layer []neuron.Neuron

func newLayer(qn int, nc neuron.NeuralConstructor) (nl *Layer) {
	l := Layer{}
	for i := 0; i < qn; i++ {
		l = append(l, *nc())
	}
	return &l
}

func NewNilLayer(qn int) (nl *Layer) {
	return newLayer(qn, neuron.NewNilNeuron)
}

func NewSigmoidLayer(qn int) (nl *Layer) {
	return newLayer(qn, neuron.NewSigmoidNeuron)
}

func NewRectifiedLinearUnitLayer(qn int) (nl *Layer) {
	return newLayer(qn, neuron.NewRectifiedLinearUnitNeuron)
}

func NewPositiveBinaryLayer(qn int) (nl *Layer) {
	return newLayer(qn, neuron.NewPositiveBinaryNeuron)
}
