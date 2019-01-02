package layer

import "jnet/pkg/neuron"

type Layer []neuron.Neuron

func New(qn int, t string) (nl *Layer) {
	l := Layer{}
	for i := 0; i < qn; i++ {
		l = append(l, *neuron.New(t))
	}
	return &l
}
