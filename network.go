package jnet

import (
	"fmt"
	"github.com/TheDemx27/calculus"
	"math"
)

type Network struct {
	layers []*layer
}

func NewNetwork(qnils []int, labels []string) (nnw *Network) {
	nnw = &Network{}

	qqnils := len(qnils)
	for qnili := 0; qnili < qqnils; qnili++ {
		qnil := qnils[qnili]
		ql := len(nnw.layers)

		var pl *layer = nil
		if ql > 0 {
			pl = nnw.layers[ql-1]
		}
		nnw.layers = append(nnw.layers, newLayer(qnil, pl))
	}

	ql := len(nnw.layers)
	lli := ql - 1
	nnw.layers[lli].setNeuronLabels(labels)

	return nnw
}

func (nw *Network) resetForPass() {
	ql := len(nw.layers)
	for li := 0; li < ql; li++ {
		l := nw.layers[li]

		l.resetForPass()
	}
}

func (nw *Network) resetForMiniBatch() {
	nw.resetForPass()

	ql := len(nw.layers)
	for li := 0; li < ql; li++ {
		l := nw.layers[li]

		l.resetForMiniBatch()
	}
}

func (nw *Network) Predict(input []float64) (prediction string) {
	nw.resetForPass()
	nw.forwardPass(input)

	return fmt.Sprintf("Prediction: %v\n", nw.getHighestConfidenceNeuron().label)
}

func (nw *Network) forwardPass(input []float64) {
	nw.layers[0].setNeuronValues(input)

	ql := len(nw.layers)
	for li := 1; li < ql; li++ { // For every layer except the first, starting from the second...
		l := nw.layers[li]

		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := l.neurons[ni]

			qc := len(n.connections)
			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.connections[ci]

				n.wSum += c.left.value * c.weight
			}

			z := n.wSum + n.bias // z_j^L
			n.value = sigmoid(z) // a_j^L
			n.dValueDNet = calculus.Diff(sigmoid, z)
			n.dNetDBias = 1.0

			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.connections[ci]

				c.dNetDWeight = c.left.value
				c.dNetDPrevValue = c.weight
			}
		}
	}
}

func (nw *Network) Train(trainingData TrainingData, trainConfig TrainingConfiguration) {
	fmt.Println("Starting training process...")

	totalLoss, averageLoss, minMiniBatchLoss, maxMiniBatchLoss := 0.0, 0.0, float64(math.MaxInt32), float64(-math.MaxInt32)

	for ti := 0; ti < trainConfig.TrainingIterations; ti++ { // For every desired training iteration...
		miniBatch := trainingData.miniBatch(trainConfig.MiniBatchSize)

		totalMiniBatchLoss := 0.0

		nw.resetForMiniBatch()
		for mbi := 0; mbi < trainConfig.MiniBatchSize; mbi++ {
			td := &miniBatch[mbi]

			nw.resetForPass()
			nw.forwardPass(td.Data)
			totalMiniBatchLoss += nw.calculateLoss(td.Truth)
			nw.backwardPass(td.Truth)
			nw.recordNudges()
		}

		miniBatchLoss := totalMiniBatchLoss / float64(trainConfig.MiniBatchSize) // Get the average loss across the whole mini batch.
		fmt.Printf("%3f ", miniBatchLoss)

		totalLoss += miniBatchLoss
		averageLoss = totalLoss / float64(ti)

		if miniBatchLoss > maxMiniBatchLoss {
			maxMiniBatchLoss = miniBatchLoss
		}
		if miniBatchLoss < minMiniBatchLoss {
			minMiniBatchLoss = miniBatchLoss
		}
		if (ti+1)%15 == 0 {
			fmt.Printf(" | %5f %5f %5f\n", averageLoss, minMiniBatchLoss, maxMiniBatchLoss)
		}

		if averageLoss < trainConfig.AverageLossCutoff {
			fmt.Printf("\nReached average loss cutoff limit, ending training process...\n")
			break
		}

		nw.calculateAverageNudges()
		nw.adjustWeights(trainConfig.LearningRate)
	}

	fmt.Println("Training process ended.")
}

func (nw *Network) backwardPass(truth []float64) {
	ll := nw.layers[len(nw.layers)-1]
	qlln := len(ll.neurons)
	for llni := 0; llni < qlln; llni++ { // For every neuron in the last layer...
		lln := ll.neurons[llni]

		lln.dLossDValue = 2 * (lln.value - truth[llni]) // d(MSE)
		lln.dLossDBias = lln.dLossDValue * lln.dValueDNet * lln.dNetDBias
		qc := len(lln.connections)
		for ci := 0; ci < qc; ci++ { // For every connection from this layer to its previous layer's neurons...
			c := lln.connections[ci]
			c.dLossDWeight = lln.dLossDValue * lln.dValueDNet * c.dNetDWeight
		}
	}

	ql := len(nw.layers)
	fli := 0
	for li := ql - 2; li >= fli; li-- { // For every layer except the last, starting from the second to last...
		l := nw.layers[li]
		nli := li + 1
		nl := nw.layers[nli]

		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := l.neurons[ni]

			qnln := len(nl.neurons)
			for nlni := 0; nlni < qnln; nlni++ { // For every neuron in the next layer...
				nln := nl.neurons[nlni]

				n.dLossDValue += nln.dLossDValue * nln.dValueDNet * nln.connections[ni].dNetDPrevValue
			}

			n.dLossDBias = n.dLossDValue * n.dValueDNet * n.dNetDBias

			qc := len(n.connections)
			for ci := 0; ci < qc; ci++ { // For every connection from this neuron to its previous layer's neurons...
				c := n.connections[ci]
				c.dLossDWeight = n.dLossDValue * n.dValueDNet * c.dNetDWeight
			}
		}
	}
}

func (nw *Network) recordNudges() {
	ql := len(nw.layers)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		l.recordNudges()
	}
}

func (nw *Network) calculateAverageNudges() {
	ql := len(nw.layers)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		l.calculateAverageNudges()
	}
}

func (nw *Network) adjustWeights(learningRate float64) {
	ql := len(nw.layers)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		l.adjustWeights(learningRate)
	}
}

func (nw *Network) calculateLoss(truth []float64) (loss float64) {
	ll := nw.layers[len(nw.layers)-1]
	qt, qn := len(truth), len(ll.neurons)

	if qt != qn {
		panic("Can't calculate loss, truth and output layer are of different lengths!")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in the last layer...
		n := ll.neurons[ni]
		loss += math.Pow(n.value-truth[ni], 2) // MSE
	}

	return loss
}

func (nw *Network) getHighestConfidenceNeuron() (hcn *neuron) {
	lli := len(nw.layers) - 1
	ll := nw.layers[lli]

	hcn = nil
	highestConfidence := float64(-math.MaxInt32)

	qlln := len(ll.neurons)
	for ni := 0; ni < qlln; ni++ { // For every neuron in the last layer...
		n := ll.neurons[ni]

		if n.value > highestConfidence {
			highestConfidence = n.value
			hcn = n
		}
	}

	return hcn
}

func (nw *Network) GetResults() (results string) {
	lli := len(nw.layers) - 1
	ll := nw.layers[lli]

	qlln := len(ll.neurons)
	for ni := 0; ni < qlln; ni++ { // For every neuron in the last layer...
		n := ll.neurons[ni]

		results = fmt.Sprintf("%v%v: %v\n", results, n.label, n.value)
	}

	hcn := nw.getHighestConfidenceNeuron()

	return fmt.Sprintf("%vResult: %v - %v\n", results, hcn.label, hcn.value)
}
