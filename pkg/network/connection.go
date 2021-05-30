package network

import (
	"math/rand"
)

// Connection represents a line of communication between the output of one
// Neuron to the input of another. Every Neuron has a set of Connections to
// every Neuron in the previous Layer. Connections are one way structures. They
// only record the connected relationship of the owning Neuron "to" a Neuron in
// the previous Layer, but not the other way around.
type Connection struct {
	// To is the neuron that this Connection's owning neuron is considered to be
	// connected "to". This neuron should be in the layer previous to the
	// Connection's owning neuron.
	To *Neuron

	// weight is the weight value for this Connection. It is a learned value
	// that emerges from the training process
	weight float64

	// dNetDWeight is the effect this Connection's weight has on the weighted
	// sum + bias. This value is calculated during a pass and lost during a
	// reset. It is solely to aid in the training process.
	dNetDWeight float64
	// dLossDWeight is the effect this Connection's weight has on the loss. This
	// value is calculated during a pass and lost during a reset. It is solely
	// to aid in the training process.
	dLossDWeight float64
	// dNetDPreValue is the effect this Connection's connected neuron's
	// activation has on the weighted sum + bias. This value is calculated
	// during a pass and lost during a reset. It is solely to aid in the
	// training process.
	dNetDPrevValue float64

	// weightNudges is the set of nudges that would push this Connection's
	// weight towards having a value which more effectively reduces the output
	// of the loss function. The average value across this slice is the
	// direction of greatest descent and is used when gradient descent is
	// executed. These values are (and should be) lost when resetFromBatch is
	// called.
	weightNudges []float64
}

// NewConnection creates a new connection assigning To to pn, which should be a
// Neuron from the previous Layer of the Network relative to the Layer the
// owning Neuron is in. The weight of this Connection is randomized.
func NewConnection(pn *Neuron) *Connection {
	return &Connection{
		To:     pn,
		weight: rand.Float64()*2 - 1, // Initialize randomly to [-1, 1)
	}
}

// resetFromBatch will reset the internal state of c from the perspective that a
// full batch/minibatch of training data was just executed. This consists of
// resetting all calculus values and the weight nudges of c.
//
// This function should always be called after running a full batch/minibatch
// before you run another one.
//
// This function differs from resetFromPass in that it clears c.weightNudges
// while resetFromPass does not.
func (c *Connection) resetFromBatch() {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0

	c.weightNudges = nil
}

// resetFromPass will reset the internal state of c from the perspective that a
// full forward and backward pass just executed. This consists of resetting all
// calculus values of c.
//
// This function should always be called after running a forward + backward pass
// before you run another one.
//
// This function differs from resetFromBatch in that it does not clear
// c.weightNudges while resetFromBatch does.
func (c *Connection) resetFromPass() {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0
}

// recordNudge records the current value of dLossDWeight as a new entry in
// c.weightNudges. After a forward + backward pass, dLossDWeight is set to the
// direction of greatest improvement of the loss function if it were to be
// applied to c.weight. It should be recorded so that when c.adjustWeights is
// called, it can be applied to reduce the loss function.
func (c *Connection) recordNudge() {
	c.weightNudges = append(c.weightNudges, c.dLossDWeight)
}

// averageWeightNudge determines the average value across all values in
// c.weightNudges.
func (c *Connection) averageWeightNudge() float64 {
	sum := 0.0
	for _, wn := range c.weightNudges {
		sum += wn
	}
	return sum / float64(len(c.weightNudges))
}

// adjustWeight adjusts c's weight in the direction of greatest improvement of
// the loss function by taking the average weight nudge, scaling it by
// learningRate, then subtracting it from the current weight value of c. The net
// impact of this is an improvement in performance against the training data
// used on this Connection.
func (c *Connection) adjustWeight(learningRate float64) {
	c.weight -= c.averageWeightNudge() * learningRate
}
