package jnet

type layer struct {
	neurons []*neuron
}

func newLayer(qn int, pl *layer) (nl *layer) {
	nl = &layer{}

	for ni := 0; ni < qn; ni++ { // For every desired neuron...
		nl.neurons = append(nl.neurons, newNeuron(pl))
	}
	return nl
}

func (l *layer) setNeuronValues(values []float64) {
	qn := len(l.neurons)

	if qn != len(values) {
		panic("Invalid number of values provided, does no match number of neurons in layer.")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.neurons[ni].value = values[ni]
	}
}

func (l *layer) setNeuronLabels(labels []string) {
	qn := len(l.neurons)

	if qn != len(labels) {
		panic("Invalid number of labels provided, does no match number of neurons in layer.")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.neurons[ni].label = labels[ni]
	}
}

func (l *layer) resetForPass() {
	qn := len(l.neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.neurons[ni]

		n.resetForPass()
	}
}

func (l *layer) resetForMiniBatch() {
	l.resetForPass()

	qn := len(l.neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.neurons[ni]

		n.resetForMiniBatch()
	}
}

func (l *layer) recordNudges() {
	qn := len(l.neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.neurons[ni]

		n.recordNudge()
	}
}

func (l *layer) calculateAverageNudges() {
	qn := len(l.neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.neurons[ni]

		n.calculateAverageNudge()
	}
}

func (l *layer) adjustWeights(learningRate float64) {
	qn := len(l.neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.neurons[ni]

		n.adjustWeights(learningRate)
	}
}
