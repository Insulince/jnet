package network

import (
	"math/rand"
)

type Connection struct {
	To *Neuron // The neuron that this Connection's owning neuron is considered to be connected "to". This neuron should be in the layer previous to the Connection's owning neuron.

	weight float64

	dNetDWeight    float64 // The effect this Connection's weight has on the weighted sum + bias.
	dLossDWeight   float64 // The effect this Connection's weight has on the loss.
	dNetDPrevValue float64 // The effect this Connection's connected neuron's activation has on the weighted sum + bias.

	weightNudges []float64
}

func NewConnection(pn *Neuron) *Connection {
	return &Connection{
		To:     pn,
		weight: rand.Float64()*2 - 1, // Initialize randomly to [-1, 1)
	}
}

func (c *Connection) resetForPass(andBatch bool) {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0

	if andBatch {
		c.weightNudges = c.weightNudges[:0]
	}
}

func (c *Connection) recordNudge() {
	c.weightNudges = append(c.weightNudges, c.dLossDWeight)
}

func (c *Connection) adjustWeight(learningRate float64) {
	c.weight -= c.averageWeightNudge() * learningRate
}

func (c *Connection) averageWeightNudge() float64 {
	sum := 0.0
	for _, wn := range c.weightNudges {
		sum += wn
	}
	return sum / float64(len(c.weightNudges))
}
