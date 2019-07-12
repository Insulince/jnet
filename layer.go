package jnet

import (
	"errors"
	"math/rand"
)

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

func (l *layer) setNeuronValues(values []float64) (err error) {
	qn := len(l.neurons)

	if qn != len(values) {
		return errors.New("invalid number of values provided, does no match number of neurons in layer")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.neurons[ni].value = values[ni]
	}

	return nil
}

func (l *layer) setNeuronLabels(labels []string) (err error) {
	qn := len(l.neurons)

	if qn != len(labels) {
		return errors.New("invalid number of labels provided, does no match number of neurons in layer")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.neurons[ni].label = labels[ni]
	}

	return nil
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

func (l *layer) mutate() {
	l.neurons[rand.Intn(len(l.neurons))].mutate()
}
