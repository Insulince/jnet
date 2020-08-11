package network

import (
	"fmt"
)

type Layer []*Neuron

func NewLayer(qn int, pl Layer) Layer {
	l := Layer{}
	for ni := 0; ni < qn; ni++ { // For every desired Neuron...
		l = append(l, NewNeuron(pl))
	}
	return l
}

func (l Layer) ConnectTo(pl Layer) {
	for ni := range l {
		l[ni].ConnectTo(pl)
	}
}

func (l Layer) SetNeuronValues(values []float64) error {
	if len(l) != len(values) {
		return fmt.Errorf("invalid number of values provided (%v), does not match number of neurons in layer (%v)", len(values), len(l))
	}

	for ni := range l {
		l[ni].SetValue(values[ni])
		l[ni].value = values[ni]
	}

	return nil
}

func (l Layer) MustSetNeuronValues(values []float64) {
	err := l.SetNeuronValues(values)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeuronLabels(labels []string) error {
	if len(l) != len(labels) {
		return fmt.Errorf("invalid number of labels provided (%v), does not match number of neurons in Layer (%v)", len(labels), len(l))
	}

	for ni := range l {
		l[ni].SetLabel(labels[ni])
	}

	return nil
}

func (l Layer) MustSetNeuronLabels(labels []string) {
	err := l.SetNeuronLabels(labels)
	if err != nil {
		panic(err)
	}
}

func (l Layer) resetForPass(andBatch bool) {
	for ni := range l {
		l[ni].resetForPass(andBatch)
	}
}

func (l Layer) recordNudges() {
	for ni := range l {
		l[ni].recordNudge()
	}
}

func (l Layer) adjustWeights(learningRate float64) {
	for ni := range l {
		l[ni].adjustWeights(learningRate)
	}
}
