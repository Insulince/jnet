package jnet

type Layer []Neuron

func newLayer(qn int, nc NeuralConstructor) (nl *Layer) {
	l := Layer{}
	for i := 0; i < qn; i++ {
		l = append(l, *nc())
	}
	return &l
}

func NewNilLayer(qn int) (nl *Layer) {
	return newLayer(qn, NewNilNeuron)
}

func NewSigmoidLayer(qn int) (nl *Layer) {
	return newLayer(qn, NewSigmoidNeuron)
}

func NewRectifiedLinearUnitLayer(qn int) (nl *Layer) {
	return newLayer(qn, NewRectifiedLinearUnitNeuron)
}

func NewPositiveBinaryLayer(qn int) (nl *Layer) {
	return newLayer(qn, NewPositiveBinaryNeuron)
}
