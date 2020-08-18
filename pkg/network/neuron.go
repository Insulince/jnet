package network

import (
	"fmt"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"math/rand"
)

type Neuron struct {
	Connections []*Connection

	ActivationFunctionName activationfunction.Name
	activationFunction     activationfunction.ActivationFunction

	label string
	value float64
	wSum  float64
	bias  float64

	dLossDValue float64 // The effect this Neuron's value has on the loss.
	dLossDBias  float64 // The effect this Neuron's bias has on the loss.
	dValueDNet  float64 // The effect this Neuron's weighted sum + bias has on the Neuron's value.
	dNetDBias   float64 // The effect this Neuron's bias has on the weighted sum + bias.

	biasNudges []float64
}

func NewNeuron(pl Layer, activationFunctionName activationfunction.Name) (*Neuron, error) {
	n := Neuron{
		bias: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
	n.ConnectTo(pl)
	err := n.SetActivationFunction(activationFunctionName)
	if err != nil {
		return nil, err
	}
	return &n, nil
}

func MustNewNeuron(pl Layer, activationFunctionName activationfunction.Name) *Neuron {
	n, err := NewNeuron(pl, activationFunctionName)
	if err != nil {
		panic(err)
	}
	return n
}

func (n *Neuron) SetLabel(label string) {
	n.label = label
}

func (n *Neuron) SetValue(value float64) {
	n.value = value
}

func (n *Neuron) SetBias(bias float64) {
	n.bias = bias
}

func (n *Neuron) SetConnectionWeights(weights []float64) error {
	if len(weights) != len(n.Connections) {
		return fmt.Errorf("invalid number of weights provided (%v), does not match number of connections in neuron (%v)", len(weights), len(n.Connections))
	}

	for ni := range n.Connections {
		n.Connections[ni].SetWeight(weights[ni])
	}
	return nil
}

func (n *Neuron) MustSetConnectionWeights(weights []float64) {
	err := n.SetConnectionWeights(weights)
	if err != nil {
		panic(err)
	}
}

// ConnectTo connects n to all the neurons in pl using brand new connections.
func (n *Neuron) ConnectTo(pl Layer) {
	n.Connections = nil
	for pni := range pl {
		n.Connections = append(n.Connections, NewConnection(pl[pni]))
	}
}

// ConnectWith connects n to all neurons in pl using the provided connections. You must provide the same number of
// connections are there are neurons in the previous layer.
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

// ConnectNeurons connects n to all neurons in pl using the existing connections. It only updates what n.Connection.To
// points to.
func (n *Neuron) ConnectNeurons(pl Layer) error {
	if len(n.Connections) != len(pl) {
		return fmt.Errorf("cannot connect neuron with previous layer using existing connections: number of existing connections (%v) does not match number of neurons in previous layer (%v)", len(n.Connections), len(pl))
	}

	for ni := range n.Connections {
		n.Connections[ni].To = pl[ni]
	}
	return nil
}

func (n *Neuron) resetForPass(andBatch bool) {
	n.value = 0.0
	n.wSum = 0.0

	n.dLossDValue = 0.0
	n.dLossDBias = 0.0
	n.dValueDNet = 0.0
	n.dNetDBias = 0.0

	if andBatch {
		n.biasNudges = n.biasNudges[:0]
	}

	for ci := range n.Connections {
		n.Connections[ci].resetForPass(andBatch)
	}
}

func (n *Neuron) recordNudge() {
	n.biasNudges = append(n.biasNudges, n.dLossDBias)

	for ci := range n.Connections { // For every Connection from this Neuron to the previous Layer's neurons...
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

func (n *Neuron) SetActivationFunction(activationFunctionName activationfunction.Name) error {
	n.ActivationFunctionName = activationFunctionName

	af, err := activationfunction.GetFunction(activationFunctionName)
	if err != nil {
		return err
	}
	n.activationFunction = af

	return nil
}

func (n *Neuron) MustSetActivationFunction(activationFunctionName activationfunction.Name) {
	err := n.SetActivationFunction(activationFunctionName)
	if err != nil {
		panic(err)
	}
}
