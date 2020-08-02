package network

import (
	"fmt"
	"math/rand"
)

type Neuron struct {
	Connections []*Connection

	value float64
	wSum  float64
	bias  float64
	label string

	dLossDValue float64 // The effect this Neuron's value has on the loss.
	dLossDBias  float64 // The effect this Neuron's bias has on the loss.
	dValueDNet  float64 // The effect this Neuron's weighted sum + bias has on the Neuron's value.
	dNetDBias   float64 // The effect this Neuron's bias has on the weighted sum + bias.

	biasNudges []float64
}

func newNeuron(pl Layer) *Neuron {
	n := Neuron{
		bias: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
	n.ConnectTo(pl)
	return &n
}

func (n *Neuron) FirstConnection() *Connection {
	return n.Connections[0]
}

func (n *Neuron) LastConnection() *Connection {
	return n.Connections[len(n.Connections)-1]
}

func (n *Neuron) GetConnection(i int) (*Connection, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get Connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return nil, fmt.Errorf("cannot get Connection at index > size of Neuron, %v (requested %v)", len(n.Connections), i)
	}
	return n.Connections[i], nil
}

func (n *Neuron) MustGetConnection(i int) *Connection {
	c, err := n.GetConnection(i)
	if err != nil {
		panic(err)
	}
	return c
}

func (n *Neuron) GetConnections(i, j int) ([]*Connection, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return nil, fmt.Errorf("cannot get subset ending at > size of connections (requested j: %v)", j)
	}
	return n.Connections[i:j], nil
}

func (n *Neuron) MustGetConnections(i, j int) []*Connection {
	ons, err := n.GetConnections(i, j)
	if err != nil {
		panic(err)
	}
	return ons
}

func (n *Neuron) SetConnection(i int, c *Connection) error {
	if i < 0 {
		return fmt.Errorf("cannot set Connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return fmt.Errorf("cannot set Connection at index > size of connections, %v (requested %v)", len(n.Connections), i)
	}
	n.Connections[i] = c
	return nil
}

func (n *Neuron) MustSetConnection(i int, c *Connection) {
	err := n.SetConnection(i, c)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SetConnections(i, j int, cs []*Connection) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return fmt.Errorf("cannot set subset ending at > size of connections (requested j: %v)", j)
	}
	q := j - i
	if len(cs) != q {
		return fmt.Errorf("cannot set a subset of Connections to a set of Connections of different length. target subset length: %v, provided set length: %v", q, len(cs))
	}
	for k := 0; k < q; k++ {
		n.Connections[k+i] = cs[k]
	}
	return nil
}

func (n *Neuron) MustSetConnections(i, j int, cs []*Connection) {
	err := n.SetConnections(i, j, cs)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SwapConnection(i int, c *Connection) (*Connection, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return nil, fmt.Errorf("cannot swap connection at index > size of connections, %v (requested %v)", len(n.Connections), i)
	}
	on := n.MustGetConnection(i)
	n.MustSetConnection(i, c)
	return on, nil
}

func (n *Neuron) MustSwapConnection(i int, c *Connection) *Connection {
	oc, err := n.SwapConnection(i, c)
	if err != nil {
		panic(err)
	}
	return oc
}

func (n *Neuron) SwapConnections(i, j int, cs []*Connection) ([]*Connection, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of connections (requested j: %v)", j)
	}
	q := j - i
	if len(cs) != q {
		return nil, fmt.Errorf("cannot swap a subset of Connections with a set of Connections of different length. target subset length: %v, provided set length: %v", q, len(cs))
	}
	ols := n.MustGetConnections(i, j)
	n.MustSetConnections(i, j, cs)
	return ols, nil
}

func (n *Neuron) MustSwapConnections(i, j int, cs []*Connection) []*Connection {
	ons, err := n.SwapConnections(i, j, cs)
	if err != nil {
		panic(err)
	}
	return ons
}

func (n *Neuron) ConnectTo(pl Layer) {
	n.Connections = nil
	for ni := range pl {
		n.Connections = append(n.Connections, newConnection(pl[ni]))
	}
}

func (n *Neuron) resetForPass() {
	n.value = 0.0
	n.wSum = 0.0

	n.dLossDValue = 0.0
	n.dLossDBias = 0.0
	n.dValueDNet = 0.0
	n.dNetDBias = 0.0

	for ci := range n.Connections {
		n.Connections[ci].resetForPass()
	}
}

func (n *Neuron) resetForMiniBatch() {
	n.value = 0.0
	n.wSum = 0.0

	n.dLossDValue = 0.0
	n.dLossDBias = 0.0
	n.dValueDNet = 0.0
	n.dNetDBias = 0.0

	n.biasNudges = n.biasNudges[:0]

	for ci := range n.Connections {
		n.Connections[ci].resetForMiniBatch()
	}
}

func (n *Neuron) recordNudge() {
	n.biasNudges = append(n.biasNudges, n.dLossDBias)

	for ci := range n.Connections { // For every Connection from this Neuron to the previous Layer's neurons...
		n.Connections[ci].recordNudge()
	}
}

func (n *Neuron) averageBiasNudge() float64 {
	var sum float64
	for _, bn := range n.biasNudges {
		sum += bn
	}
	return sum / float64(len(n.biasNudges))
}

func (n *Neuron) adjustWeights(learningRate float64) {
	n.bias -= n.averageBiasNudge() * learningRate

	for ci := range n.Connections { // For every Connection from this Neuron to the previous Layer's neurons...
		n.Connections[ci].adjustWeight(learningRate)
	}
}
