package network

import (
	"math/rand"
)

type Connection struct {
	left *Neuron // TODO(justin): rename

	weight float64

	dNetDWeight    float64 // The effect this Connection's weight has on the weighted sum + bias.
	dLossDWeight   float64 // The effect this Connection's weight has on the loss.
	dNetDPrevValue float64 // The effect this Connection's left-Neuron's activation has on the weighted sum + bias.

	weightNudges []float64
}

func newConnection(left *Neuron) *Connection {
	return &Connection{
		left:   left,
		weight: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
}

func (c *Connection) resetForPass() {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0
}

func (c *Connection) resetForMiniBatch() {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0

	c.weightNudges = c.weightNudges[:0]
}

func (c *Connection) recordNudge() {
	c.weightNudges = append(c.weightNudges, c.dLossDWeight)
}

func (c *Connection) averageWeightNudge() float64 {
	var sum float64
	for _, nudge := range c.weightNudges {
		sum += nudge
	}
	return sum / float64(len(c.weightNudges))
}

func (c *Connection) adjustWeight(learningRate float64) {
	c.weight -= c.averageWeightNudge() * learningRate
}
