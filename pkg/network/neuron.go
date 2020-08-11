package network

import (
	"math/rand"
)

type Neuron struct {
	Connections []*Connection

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

func NewNeuron(pl Layer) *Neuron {
	n := Neuron{
		bias: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
	n.ConnectTo(pl)
	return &n
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

func (n *Neuron) ConnectTo(pl Layer) {
	n.Connections = nil
	for ni := range pl {
		n.Connections = append(n.Connections, NewConnection(pl[ni]))
	}
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
