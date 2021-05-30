package network

import (
	"fmt"
	"math/rand"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
)

type Neuron struct {
	Connections []*Connection

	ActivationFunctionName activationfunction.Name
	activationFunction     activationfunction.ActivationFunction

	label string
	value float64
	bias  float64

	wSum float64

	// The effect this Neuron's value has on the loss.
	dLossDValue float64
	// The effect this Neuron's bias has on the loss.
	dLossDBias float64
	// The effect this Neuron's weighted sum + bias has on the Neuron's value.
	dValueDNet float64
	// The effect this Neuron's bias has on the weighted sum + bias.
	dNetDBias float64

	biasNudges []float64
}

func NewNeuron(pl Layer, activationFunctionName activationfunction.Name) (*Neuron, error) {
	n := Neuron{
		bias: rand.Float64()*2 - 1, // Initialize randomly to [-1, 1)
	}
	n.ConnectTo(pl)
	err := n.SetActivationFunction(activationFunctionName)
	if err != nil {
		return nil, err
	}
	return &n, nil
}

// MustNewNeuron calls NewNeuron but panics if an error is encountered.
func MustNewNeuron(pl Layer, activationFunctionName activationfunction.Name) *Neuron {
	n, err := NewNeuron(pl, activationFunctionName)
	if err != nil {
		panic(err)
	}
	return n
}

// ConnectTo connects n to all the neurons in pl using brand new connections.
func (n *Neuron) ConnectTo(pl Layer) {
	n.Connections = nil
	for pni := range pl {
		n.Connections = append(n.Connections, NewConnection(pl[pni]))
	}
}

// ConnectWith connects n to all neurons in pl using the provided connections.
// You must provide the same number of connections are there are neurons in the
// previous layer.
func (n *Neuron) ConnectWith(pl Layer, pcs []*Connection) error {
	if len(pcs) != len(pl) {
		return fmt.Errorf("cannot connect neuron with layer using provided connections: number of connections (%v) does not match number of neurons in previous layer (%v)", len(pcs), len(pl))
	}

	n.Connections = nil
	for pni := range pl {
		n.Connections = append(n.Connections, pcs[pni])
	}
	return nil
}

// ConnectNeurons connects n to all neurons in pl using the existing
// connections. It only updates what n.Connection.To points to. All other values
// are preserved.
func (n *Neuron) ConnectNeurons(pl Layer) error {
	if len(n.Connections) != len(pl) {
		return fmt.Errorf("cannot connect neuron with previous layer using existing connections: number of existing connections (%v) does not match number of neurons in previous layer (%v)", len(n.Connections), len(pl))
	}

	for ci := range n.Connections {
		n.Connections[ci].To = pl[ci]
	}
	return nil
}

func (n *Neuron) resetFromBatch() {
	n.value = 0.0
	n.wSum = 0.0

	n.dLossDValue = 0.0
	n.dLossDBias = 0.0
	n.dValueDNet = 0.0
	n.dNetDBias = 0.0

	n.biasNudges = nil

	for ci := range n.Connections {
		n.Connections[ci].resetFromBatch()
	}
}

func (n *Neuron) resetFromPass() {
	n.value = 0.0
	n.wSum = 0.0

	n.dLossDValue = 0.0
	n.dLossDBias = 0.0
	n.dValueDNet = 0.0
	n.dNetDBias = 0.0

	for ci := range n.Connections {
		n.Connections[ci].resetFromPass()
	}
}

func (n *Neuron) recordNudge() {
	n.biasNudges = append(n.biasNudges, n.dLossDBias)

	// For every Connection from this Neuron to the previous Layer's neurons...
	for ci := range n.Connections {
		n.Connections[ci].recordNudge()
	}
}

func (n *Neuron) averageBiasNudge() float64 {
	sum := 0.0
	for _, bn := range n.biasNudges {
		sum += bn
	}
	return sum / float64(len(n.biasNudges))
}

func (n *Neuron) adjustWeights(learningRate float64) {
	n.bias -= n.averageBiasNudge() * learningRate

	for ci := range n.Connections {
		n.Connections[ci].adjustWeight(learningRate)
	}
}
