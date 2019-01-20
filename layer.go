package jnet

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(qn int, pl *Layer) (nl *Layer) {
	nl = &Layer{}

	for i := 0; i < qn; i++ {
		nl.Neurons = append(nl.Neurons, NewNeuron(pl))
	}
	return nl
}

func (l *Layer) SetNeuronValues(values []float64) (this *Layer) {
	qn := len(l.Neurons)

	if qn != len(values) {
		panic("Invalid number of values provided, does no match number of neurons in layer.")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.Neurons[ni].Value = values[ni]
	}

	return l
}

func (l *Layer) resetForPass() (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ {
		n := l.Neurons[ni]

		n.ResetForPass()
	}

	return l
}

func (l *Layer) resetForMiniBatch() (this *Layer) {
	l.resetForPass()

	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ {
		n := l.Neurons[ni]

		n.ResetForMiniBatch()
	}

	return l
}

func (l *Layer) recordNudges() (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.Neurons[ni]

		n.recordNudges()
	}

	return l
}

func (l *Layer) averageNudges() (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.Neurons[ni]

		n.averageNudges()
	}

	return l
}

func (l *Layer) adjustWeights(learningRate float64) (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.Neurons[ni]

		n.adjustWeights(learningRate)
	}

	return l
}
