package network

import (
	"fmt"
	"math/rand"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
)

// Neuron represents the base unit of a neural network. Each neuron should have
// a set of connections to every neuron in the previous layer via its
// Connections field. Additionally every neuron has an activation function
// associated with it for non-linearization of the weighted inputs from the
// previous layer.
type Neuron struct {
	// Connections is the set of weighted connections from this Neuron to the
	// previous layer's neurons. Connections refer to the "owning neuron", and
	// that refers to this neuron, not the neuron this connection leads to.
	Connections []*Connection

	// ActivationFunctionName is the activationfunction.Name that corresponds to
	// the activation function this Neuron should use.
	ActivationFunctionName activationfunction.Name
	// activationFunction is the actual activation function that this Neuron
	// should use.
	activationFunction activationfunction.ActivationFunction

	// label is the string identifier used to interpret the inputs or outputs
	// that this neuron corresponds to. It should only be set on input and/or
	// output Neurons. Hidden layer neurons should not use this field.
	label string
	// value represents how much or little this Neuron fired on the last pass.
	// It is safe to interpret this field after a pass is completed, but if a
	// pass has not been completed OR a reset has been executed, then value will
	// not be meaningful.
	value float64
	// bias is the bias value for this neuron. It is a learned value that
	// emerges from the training process
	bias float64
	// wSum is the weighted sum that was fed into this neuron during the last
	// pass. It is the raw input value prior to going through
	// activationFunction.
	wSum float64

	// dLossDValue is the effect this Neuron's value has on the loss. This value
	// is calculated during a pass and lost during a reset. It is solely to aid
	// in the training process.
	dLossDValue float64
	// dLossDBias is the effect this Neuron's bias has on the loss. This value
	// is calculated during a pass and lost during a reset. It is solely to aid
	// in the training process.
	dLossDBias float64
	// dValueDNet is the effect this Neuron's weighted sum + bias has on the
	// Neuron's value. This value is calculated during a pass and lost during
	// a reset. It is solely to aid in the training process.
	dValueDNet float64
	// dNetDBias is the effect this Neuron's bias has on the weighted sum +
	// bias. This value is calculated during a pass and lost during a reset. It
	// is solely to aid in the training process.
	dNetDBias float64

	// biasNudges is the set of nudges that would push this Neuron's bias
	// towards having a value which more effectively reduces the output of the
	// loss function. The average value across this slice is the direction of
	// greatest descent and is used when gradient descent is executed. These
	// values are (and should be) lost when resetFromBatch is called.
	biasNudges []float64
}

// NewNeuron creates a new Neuron and connects it to pl with randomized weights
// and a randomized bias value. activationFunctionName is used to assign an
// activation function to this Neuron. If an activation function can't be found
// matching the provided activationfunction.Name, an error is returned.
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

// resetFromBatch will reset the internal state of n from the perspective that a
// full batch/minibatch of training data was just executed. This consists of
// resetting the value, weighted sum, all calculus values, and the bias nudges
// of n. Additionally each connection in n.Connections is reset as well.
//
// This function should always be called after running a full batch/minibatch
// before you run another one.
//
// This function differs from resetFromPass in that it clears n.biasNudges while
// resetFromPass does not.
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

// resetFromPass will reset the internal state of n from the perspective that a
// full forward and backward pass just executed. This consists of resetting the
// value, weighted sum, and all calculus values of n. Additionally each
// connection in n.Connections is reset as well.
//
// This function should always be called after running a forward + backward pass
// before you run another one.
//
// This function differs from resetFromBatch in that it does not clear
// n.biasNudges while resetFromBatch does.
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

// recordNudges records the current value of dLossDBias as a new entry in
// n.biasNudges. After a forward + backward pass, dLossDBias is set to the
// direction of greatest improvement of the loss function if it were to be
// applied to n.bias. It should be recorded so that when n.adjustWeights is
// called, it can be applied to reduce the loss function.
//
// recordNudges also records all nudges to n's Connections.
func (n *Neuron) recordNudges() {
	n.biasNudges = append(n.biasNudges, n.dLossDBias)

	// For every Connection from this Neuron to the previous Layer's neurons...
	for ci := range n.Connections {
		n.Connections[ci].recordNudge()
	}
}

// averageBiasNudge determines the average value across all values in
// n.biasNudges.
func (n *Neuron) averageBiasNudge() float64 {
	sum := 0.0
	for _, bn := range n.biasNudges {
		sum += bn
	}
	return sum / float64(len(n.biasNudges))
}

// adjustWeights adjusts n's bias in the direction of greatest improvement of
// the loss function by taking the average bias nudge, scaling it by
// learningRate, then subtracting it from the current bias value of n. The net
// impact of this is an improvement in performance against the training data
// used on this Neuron.
//
// adjustWeights also adjusts all weights in n's Connections.
func (n *Neuron) adjustWeights(learningRate float64) {
	n.bias -= n.averageBiasNudge() * learningRate

	for ci := range n.Connections {
		n.Connections[ci].adjustWeight(learningRate)
	}
}
