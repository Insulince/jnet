package network

import (
	"fmt"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
)

// Layer is an isolated set of neurons in the network that constitute a single
// layer. A layer can be input, output, or hidden, a distinction which is only
// recognized by its position in the Network.
type Layer []*Neuron

// NewLayer will construct a new Layer with qn neurons each of which with the
// activation function corresponding to activationFunctionName. The new layer
// will have its new neurons connected to all neurons in pl, which should be the
// previous layer in the network.
//
// If the network to be created is the input layer and does not have a previous
// layer, nil is an acceptable value to provide for pl.
func NewLayer(qn int, pl Layer, activationFunctionName activationfunction.Name) (Layer, error) {
	l := Layer{}
	for ni := 0; ni < qn; ni++ { // For every desired Neuron...
		n, err := NewNeuron(pl, activationFunctionName)
		if err != nil {
			return nil, err
		}
		l = append(l, n)
	}
	return l, nil
}

// MustNewLayer calls NewLayer but panics if an error is encountered.
func MustNewLayer(qn int, pl Layer, activationFunctionName activationfunction.Name) Layer {
	n, err := NewLayer(qn, pl, activationFunctionName)
	if err != nil {
		panic(err)
	}
	return n
}

// ConnectTo connects all the neurons in l to all the neurons in pl using brand
// new connections.
func (l Layer) ConnectTo(pl Layer) {
	for ni := range l {
		l[ni].ConnectTo(pl)
	}
}

// ConnectWith connects all the neurons in l to all the neurons in pl using the
// provided connections. cs is an MxN slice of connections where M is the number
// of neurons in l, and n is the number of neurons in pl. If this is not
// honored, an error will be returned.
func (l Layer) ConnectWith(pl Layer, cs [][]*Connection) error {
	if len(cs) != len(l) {
		return fmt.Errorf("cannot connect layer with previous layer using provided connections: number of provided sets of connections (%v) does not match number of neurons in layer (%v)", len(cs), len(l))
	}

	for ni := range l {
		err := l[ni].ConnectWith(pl, cs[ni])
		if err != nil {
			return err
		}
	}
	return nil
}

// ConnectNeurons connects all neurons in l to all neurons in pl using the
// existing connections. It only updates what each neurons Connection.To points
// to. All other values are preserved.
func (l Layer) ConnectNeurons(pl Layer) error {
	for ni := range l {
		err := l[ni].ConnectNeurons(pl)
		if err != nil {
			return err
		}
	}
	return nil
}

func (l Layer) resetFromBatch() {
	for ni := range l {
		l[ni].resetFromBatch()
	}
}

func (l Layer) resetFromPass() {
	for ni := range l {
		l[ni].resetFromPass()
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
