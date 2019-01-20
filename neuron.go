package jnet

import "math/rand"

type Neuron struct {
	Value       float64
	WSum        float64
	Bias        float64
	Connections []*Connection

	dLossDValue float64 // The effect this neuron's value has on the loss (Calculated in back prop).
	dLossDBias  float64 // The effect this neuron's bias has on the loss (Calculated in back prop). = dLossDValue * dValueDNet * dNetDBias = dLossDValue * dValueDNet
	dValueDNet  float64 // The effect this neuron's weighted sum + bias has on the neuron's value (Calculated in forward pass).
	dNetDBias   float64 // The effect this neuron's bias has on the weighted sum + bias (Calculated in forward pass). (Always = 1.0)

	BiasNudges       []float64
	AverageBiasNudge float64
}

func NewNeuron(pl *Layer) (nn *Neuron) {
	nn = &Neuron{
		Bias: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}

	if pl != nil {
		qpln := len(pl.Neurons)
		for ni := 0; ni < qpln; ni++ { // For every neuron in the previous layer...
			pln := pl.Neurons[ni]

			nn.Connections = append(nn.Connections, NewConnection(pln, nn))
		}
	} else {
		nn.Bias = 0.0 // This is an input neuron, so set the bias to 0.0 to minimize any confusion when debugging.
	}

	return nn
}

func (n *Neuron) ResetForPass() (this *Neuron) {
	n.Value = 0.0
	n.WSum = 0.0

	n.dLossDValue = 0.0
	n.dLossDBias = 0.0
	n.dValueDNet = 0.0
	n.dNetDBias = 0.0

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.ResetForPass()
	}

	return n
}

func (n *Neuron) ResetForMiniBatch() (this *Neuron) {
	n.ResetForPass()

	n.BiasNudges = []float64{}
	n.AverageBiasNudge = 0.0

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.resetForMiniBatch()
	}

	return n
}

func (n *Neuron) recordNudges() (this *Neuron) {
	n.BiasNudges = append(n.BiasNudges, n.dLossDBias)

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.recordNudges()
	}

	return n
}

func (n *Neuron) averageNudges() (this *Neuron) {
	dcdbSum := 0.0
	for _, dcdb := range n.BiasNudges {
		dcdbSum += dcdb
	}

	n.AverageBiasNudge = dcdbSum / float64(len(n.BiasNudges))

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.averageNudges()
	}

	return n
}

func (n *Neuron) adjustWeights(learningRate float64) (this *Neuron) {
	n.Bias -= n.AverageBiasNudge * learningRate

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.adjustWeights(learningRate)
	}

	return n
}
