package layer

import (
	"jnet/pkg/connection"
	"jnet/pkg/data"
	"jnet/pkg/neuron"
)

type Layer []neuron.Neuron

func New(qn int, t string) (nl *Layer) {
	l := Layer{}
	for ni := 0; ni < qn; ni++ { // For every neuron requested...
		l = append(l, *neuron.New(t))
	}

	return &l
}

func (l *Layer) SetInputNeuronValues(values data.D) (this *Layer) {
	if len(*l) != len(values) {
		panic("Invalid number of inputs provided, does no match number of input neurons.")
	}

	qn := len(*l)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.SetInputNeuronValue(ni, values[ni])
	}

	return l
}

func (l *Layer) SetInputNeuronValue(ni int, v data.V) (this *Layer) {
	(*l)[ni].Value = v

	return l
}

func (l *Layer) SetConnectionWeights(cm connection.Map) (this *Layer) {
	qc := len(cm)
	for ci := 0; ci < qc; ci++ { // For every connection in the connection map...
		c := cm[ci]
		l.SetConnectionWeight(c)
	}

	return l
}

func (l *Layer) SetConnectionWeight(cm connection.Connection) (this *Layer) {
	ni, ci, w := int(cm[connection.IndexFrom]), int(cm[connection.IndexTo]), cm[connection.IndexWeight]

	(*l)[ni].Connections[ci][connection.IndexWeight] = w

	return l
}

func (l *Layer) SetOutputNeuronResults(outputs []string) (this *Layer) {
	if len(*l) != len(outputs) {
		panic("Invalid number of outputs provided, does no match number of output neurons.")
	}

	qn := len(*l)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.SetOutputNeuronResult(ni, outputs[ni])
	}

	return l
}

func (l *Layer) SetOutputNeuronResult(ni int, r string) (this *Layer) {
	(*l)[ni].Result = r

	return l
}
