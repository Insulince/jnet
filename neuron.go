package jnet

import "math/rand"

type neuron struct {
	value       float64
	wSum        float64
	bias        float64
	connections []*connection
	label       string

	dLossDValue float64 // The effect this neuron's value has on the loss.
	dLossDBias  float64 // The effect this neuron's bias has on the loss.
	dValueDNet  float64 // The effect this neuron's weighted sum + bias has on the neuron's value.
	dNetDBias   float64 // The effect this neuron's bias has on the weighted sum + bias.

	biasNudges       []float64
	averageBiasNudge float64
}

func newNeuron(pl *layer) (nn *neuron) {
	nn = &neuron{
		bias: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}

	if pl != nil {
		qpln := len(pl.neurons)
		for ni := 0; ni < qpln; ni++ { // For every neuron in the previous layer...
			pln := pl.neurons[ni]

			nn.connections = append(nn.connections, newConnection(pln, nn))
		}
	} else {
		nn.bias = 0.0 // This is an input neuron, so set the bias to 0.0 to minimize any confusion when debugging.
	}

	return nn
}

func (n *neuron) resetForPass() {
	n.value = 0.0
	n.wSum = 0.0

	n.dLossDValue = 0.0
	n.dLossDBias = 0.0
	n.dValueDNet = 0.0
	n.dNetDBias = 0.0

	qc := len(n.connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.connections[ci]

		c.resetForPass()
	}
}

func (n *neuron) resetForMiniBatch() {
	n.resetForPass()

	n.biasNudges = []float64{}
	n.averageBiasNudge = 0.0

	qc := len(n.connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.connections[ci]

		c.resetForMiniBatch()
	}
}

func (n *neuron) recordNudge() {
	n.biasNudges = append(n.biasNudges, n.dLossDBias)

	qc := len(n.connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.connections[ci]

		c.recordNudge()
	}
}

func (n *neuron) calculateAverageNudge() {
	sum := 0.0

	qbn := len(n.biasNudges)
	for bni := 0; bni < qbn; bni++ { // For every bias nudge in this neuron...
		bn := n.biasNudges[bni]

		sum += bn
	}

	n.averageBiasNudge = sum / float64(qbn)

	qc := len(n.connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.connections[ci]

		c.calculateAverageNudge()
	}
}

func (n *neuron) adjustWeights(learningRate float64) {
	n.bias -= n.averageBiasNudge * learningRate

	qc := len(n.connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.connections[ci]

		c.adjustWeight(learningRate)
	}
}

func (n *neuron) mutate() {
	r := rand.Intn(len(n.connections) + 1) // + 1 to include a chance to mutate the bias instead.
	if r != len(n.connections) {           // If this is not that +1 chance...
		n.connections[r].mutate() // Mutate a connection.
	} else { // Otherwise...
		n.bias = rand.Float64()*2 - 1 // Mutate randomly to [-1.0, 1.0) // Mutate the bias.
	}
}
