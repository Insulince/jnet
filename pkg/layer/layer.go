package layer

import (
	"jnet/pkg/connection"
	"jnet/pkg/neuron"
)

type Layer []neuron.Neuron

func New(qn int, t string) (nl *Layer) {
	l := Layer{}
	for i := 0; i < qn; i++ {
		l = append(l, *neuron.New(t))
	}
	return &l
}

func (l *Layer) SetInputNeuronValues(values []float64) (this *Layer) {
	if len(*l) != len(values) {
		panic("Invalid number of inputs provided, does no match number of input neurons.")
	}

	for ni := 0; ni < len(*l); ni++ {
		l.SetInputNeuronValue(ni, values[ni])
	}

	return l
}

func (l *Layer) SetInputNeuronValue(ni int, v float64) (this *Layer) {
	(*l)[ni].Value = v

	return l
}

func (l *Layer) SetConnectionWeights(cmc connection.Map) (this *Layer) {
	for _, cm := range cmc {
		l.SetConnectionWeight(cm)
	}

	return l
}

func (l *Layer) SetConnectionWeight(cm connection.Connection) (this *Layer) {
	(*l)[(int)(cm[connection.IndexFrom])].Connections[(int)(cm[connection.IndexTo])][connection.IndexWeight] = cm[connection.IndexWeight]

	return l
}

func (l *Layer) SetOutputNeuronResults(outputs []string) (this *Layer) {
	if len(*l) != len(outputs) {
		panic("Invalid number of outputs provided, does no match number of output neurons.")
	}

	for ni := 0; ni < len(*l); ni++ {
		l.SetOutputNeuronResult(ni, outputs[ni])
	}

	return l
}

func (l *Layer) SetOutputNeuronResult(ni int, r string) (this *Layer) {
	(*l)[ni].Result = r

	return l
}
