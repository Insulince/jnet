package jnet

import "math/rand"

type Connection struct {
	Left   *Neuron
	Weight float64
	Right  *Neuron

	dNetDWeight    float64 // The effect this connection's weight has on the weighted sum + bias. ->
	dLossDWeight   float64 // The effect this connection's weight has on the loss. = dLossDValue * dValueDNet * dNetDWeight <-
	dNetDPrevValue float64 // The effect this connection's left-neuron's activation has on the weighted sum + bias. ->

	WeightNudges       []float64
	AverageWeightNudge float64
}

func NewConnection(left *Neuron, right *Neuron) (c *Connection) {
	return &Connection{
		Left:   left,
		Right:  right,
		Weight: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
}

func (c *Connection) ResetForPass() (this *Connection) {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0

	return c
}

func (c *Connection) resetForMiniBatch() (this *Connection) {
	c.ResetForPass()

	c.WeightNudges = []float64{}
	c.AverageWeightNudge = 0.0

	return c
}

func (c *Connection) recordNudges() (this *Connection) {
	c.WeightNudges = append(c.WeightNudges, c.dLossDWeight)

	return c
}

func (c *Connection) averageNudges() (this *Connection) {
	dcdwSum := 0.0
	for _, dcdw := range c.WeightNudges {
		dcdwSum += dcdw
	}

	c.AverageWeightNudge = dcdwSum / float64(len(c.WeightNudges))

	return c
}

func (c *Connection) adjustWeights(learningRate float64) (this *Connection) {
	c.Weight -= c.AverageWeightNudge * learningRate

	return c
}
