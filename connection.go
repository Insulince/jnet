package jnet

import "math/rand"

type connection struct {
	left   *neuron
	weight float64
	right  *neuron

	dNetDWeight    float64 // The effect this connection's weight has on the weighted sum + bias. ->
	dLossDWeight   float64 // The effect this connection's weight has on the loss. = dLossDValue * dValueDNet * dNetDWeight <-
	dNetDPrevValue float64 // The effect this connection's left-neuron's activation has on the weighted sum + bias. ->

	weightNudges       []float64
	averageWeightNudge float64
}

func newConnection(left *neuron, right *neuron) (c *connection) {
	return &connection{
		left:   left,
		right:  right,
		weight: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
}

func (c *connection) resetForPass() {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0
}

func (c *connection) resetForMiniBatch() {
	c.resetForPass()

	c.weightNudges = []float64{}
	c.averageWeightNudge = 0.0
}

func (c *connection) recordNudge() {
	c.weightNudges = append(c.weightNudges, c.dLossDWeight)
}

func (c *connection) calculateAverageNudge() {
	sum := 0.0

	qwn := len(c.weightNudges)
	for wni := 0; wni < qwn; wni++ { // For every weight nudge in this connection...
		wn := c.weightNudges[wni]

		sum += wn
	}

	c.averageWeightNudge = sum / float64(qwn)
}

func (c *connection) adjustWeight(learningRate float64) {
	c.weight -= c.averageWeightNudge * learningRate
}
