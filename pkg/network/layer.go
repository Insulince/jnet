package network

import (
	"fmt"
)

type Layer []*Neuron

func newLayer(qn int, pl Layer) Layer {
	l := Layer{}
	for ni := 0; ni < qn; ni++ { // For every desired Neuron...
		l = append(l, newNeuron(pl))
	}
	return l
}

func (l Layer) FirstNeuron() *Neuron {
	return l[0]
}

func (l Layer) LastNeuron() *Neuron {
	return l[len(l)-1]
}

func (l Layer) GetNeuron(i int) (*Neuron, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return nil, fmt.Errorf("cannot get neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	return l[i], nil
}

func (l Layer) MustGetNeuron(i int) *Neuron {
	n, err := l.GetNeuron(i)
	if err != nil {
		panic(err)
	}
	return n
}

func (l Layer) GetNeurons(i, j int) ([]*Neuron, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return nil, fmt.Errorf("cannot get subset ending at > size of layer (requested j: %v)", j)
	}
	return l[i:j], nil
}

func (l Layer) MustGetNeurons(i, j int) []*Neuron {
	ons, err := l.GetNeurons(i, j)
	if err != nil {
		panic(err)
	}
	return ons
}

func (l Layer) SetNeuron(i int, n *Neuron, pl Layer) error {
	if i < 0 {
		return fmt.Errorf("cannot set neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return fmt.Errorf("cannot set neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	l[i] = n
	if i > 0 {
		n.ConnectTo(pl)
	}
	return nil
}

func (l Layer) MustSetNeuron(i int, n *Neuron, pl Layer) {
	err := l.SetNeuron(i, n, pl)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeurons(i, j int, ns []*Neuron, pl Layer) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return fmt.Errorf("cannot set subset ending at > size of layer (requested j: %v)", j)
	}
	q := j - i
	if len(ns) != q {
		return fmt.Errorf("cannot set a subset of neurons to a set of neurons of different length. target subset length: %v, provided set length: %v", q, len(ns))
	}
	for k := 0; k < q; k++ {
		l[k+i] = ns[k]
		if k+i > 0 {
			ns[k].ConnectTo(pl)
		}
	}
	return nil
}

func (l Layer) MustSetNeurons(i, j int, ns []*Neuron, pl Layer) {
	err := l.SetNeurons(i, j, ns, pl)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SwapNeuron(i int, n *Neuron, pl Layer) (*Neuron, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return nil, fmt.Errorf("cannot swap neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	on := l.MustGetNeuron(i)
	l.MustSetNeuron(i, n, pl)
	return on, nil
}

func (l Layer) MustSwapNeuron(i int, n *Neuron, pl Layer) *Neuron {
	on, err := l.SwapNeuron(i, n, pl)
	if err != nil {
		panic(err)
	}
	return on
}

func (l Layer) SwapNeurons(i, j int, ns []*Neuron, pl Layer) ([]*Neuron, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of layer (requested j: %v)", j)
	}
	q := j - i
	if len(ns) != q {
		return nil, fmt.Errorf("cannot swap a subset of neurons with a set of neurons of different length. target subset length: %v, provided set length: %v", q, len(ns))
	}
	ols := l.MustGetNeurons(i, j)
	l.MustSetNeurons(i, j, ns, pl)
	return ols, nil
}

func (l Layer) MustSwapNeurons(i, j int, ns []*Neuron, pl Layer) []*Neuron {
	ons, err := l.SwapNeurons(i, j, ns, pl)
	if err != nil {
		panic(err)
	}
	return ons
}

func (l Layer) ConnectTo(pl Layer) {
	for ni := range l {
		l[ni].ConnectTo(pl)
	}
}

func (l Layer) setNeuronValues(values []float64) error {
	if len(l) != len(values) {
		return fmt.Errorf("invalid number of values provided (%v), does not match number of neurons in layer (%v)", len(values), len(l))
	}

	for ni := range l {
		l[ni].value = values[ni]
	}

	return nil
}

func (l Layer) mustSetNeuronValues(values []float64) {
	err := l.setNeuronValues(values)
	if err != nil {
		panic(err)
	}
}

func (l Layer) setNeuronLabels(labels []string) error {
	if len(l) != len(labels) {
		return fmt.Errorf("invalid number of labels provided (%v), does not match number of neurons in Layer (%v)", len(labels), len(l))
	}

	for ni := range l {
		l[ni].label = labels[ni]
	}

	return nil
}

func (l Layer) mustSetNeuronLabels(labels []string) {
	err := l.setNeuronLabels(labels)
	if err != nil {
		panic(err)
	}
}

func (l Layer) resetForPass() {
	for ni := range l {
		l[ni].resetForPass()
	}
}

func (l Layer) resetForMiniBatch() {
	for ni := range l {
		l[ni].resetForMiniBatch()
	}
}

func (l Layer) recordNudges() {
	for ni := range l {
		l[ni].recordNudge()
	}
}

func (l Layer) adjustWeights(learningRate float64) {
	qn := len(l)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l[ni]

		n.adjustWeights(learningRate)
	}
}

func (l Layer) calculateAverageNudges() {
	qn := len(l)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l[ni]

		n.calculateAverageNudge()
	}
}
