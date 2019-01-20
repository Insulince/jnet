package jnet

import (
	"fmt"
	"github.com/TheDemx27/calculus"
	"math"
	"math/rand"
)

type Network struct {
	Layers []*Layer
	Loss   float64
}

func NewNetwork(qnils ...int) (nnw *Network) {
	nnw = &Network{}

	qqnils := len(qnils)
	for qnili := 0; qnili < qqnils; qnili++ {
		qnil := qnils[qnili]
		ql := len(nnw.Layers)

		var pl *Layer = nil
		if ql > 0 {
			pl = nnw.Layers[ql-1]
		}
		nnw.Layers = append(nnw.Layers, NewLayer(qnil, pl))
	}

	return nnw
}

func (nw *Network) resetForPass() (this *Network) {
	ql := len(nw.Layers)
	for li := 0; li < ql; li++ {
		l := nw.Layers[li]

		l.resetForPass()
	}

	return nw
}

func (nw *Network) resetForMiniBatch() (this *Network) {
	nw.resetForPass()

	ql := len(nw.Layers)
	for li := 0; li < ql; li++ {
		l := nw.Layers[li]

		l.resetForMiniBatch()
	}
	return nw
}

// Predict will run an input vector, `input` against the neural network which sets its outputs for a prediction.
// Currently it just acts as an exported wrapper around forwardPass, because there is no need
// to do anything except a forward pass, however, to stay in line with BackwardPass, I made
// forwardPass unexported.
func (nw *Network) Predict(input []float64) (this *Network) {
	nw.resetForPass()
	nw.forwardPass(input)

	return nw
}

// forwardPass runs the neural network against an input vector, `input`, and set its outputs for a prediction.
func (nw *Network) forwardPass(input []float64) (this *Network) {
	nw.Loss = 0.0
	nw.Layers[0].SetNeuronValues(input)

	ql := len(nw.Layers)
	for li := 1; li < ql; li++ { // For every layer except the first, starting from the second...
		l := nw.Layers[li]

		qn := len(l.Neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := l.Neurons[ni]

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.Connections[ci]

				n.WSum += c.Left.Value * c.Weight
			}

			z := n.WSum + n.Bias // z_j^L
			n.Value = sigmoid(z) // a_j^L
			n.dValueDNet = calculus.Diff(sigmoid, z)
			n.dNetDBias = 1.0

			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.Connections[ci]

				c.dNetDWeight = c.Left.Value
				c.dNetDPrevValue = c.Weight
			}
		}
	}

	return nw
}

// Train will exercise the network against a series of "truth" input vectors, `trainingData`, choosing one at random, `TrainingIterations` times.
// Each iteration, a forward pass is executed to get the current outputs, loss is calculated, a backward pass is executed to calculate the gradients,
// and each weight is adjusted to make the network perform better.
func (nw *Network) Train(trainingData []TrainingData, learningRate float64, trainingIterations int, miniBatchSize int) (this *Network) {
	fmt.Println("TRAINING START")

	c := 0

	totalAverageLoss := 0.0
	averageAverageLoss := 0.0
	minAverageLoss := float64(math.MaxInt32)
	maxAverageLoss := float64(-math.MaxInt32)

	// TODO: Stop this when sufficiently minimized.
	for i := 0; i < trainingIterations; i++ { // For every desired training iteration...
		// Shuffle current set of trainingData
		for i := range trainingData {
			j := rand.Intn(i + 1)
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		}

		// Slice out all trainingData elements we need.
		miniBatch := trainingData[0:miniBatchSize]

		totalLoss := 0.0

		nw.resetForMiniBatch()
		for mbi := 0; mbi < miniBatchSize; mbi++ {
			td := &miniBatch[mbi]

			nw.resetForPass()
			nw.forwardPass(td.Data)

			nw.CalculateLoss(td.Truth)
			totalLoss += nw.Loss

			nw.backwardPass(td.Truth)

			nw.recordNudges()
		}

		c++
		averageLoss := totalLoss / float64(miniBatchSize) // Get the average loss across the whole minibatch.
		fmt.Printf("%3f ", averageLoss)

		totalAverageLoss += averageLoss
		averageAverageLoss = totalAverageLoss / float64(i)
		if averageLoss > maxAverageLoss {
			maxAverageLoss = averageLoss
		}
		if averageLoss < minAverageLoss {
			minAverageLoss = averageLoss
		}

		if c > 15 {
			fmt.Printf(" | %5f %5f %5f\n", averageAverageLoss, minAverageLoss, maxAverageLoss)
			c = 0
		}

		// TODO: Make better
		if averageAverageLoss < 0.1 {
			break
		}

		nw.averageNudges()

		nw.adjustWeights(learningRate)
	}
	fmt.Println()

	return nw
}

// BackwardPass runs the network backwards from the state its last forward pass left it in to determine the LossGradients of each neuron (by
// comparing to the provided loss value, `loss`), and adjusts each neuron's weight based on it to make the network perform better.
func (nw *Network) backwardPass(truth []float64) (this *Network) {
	ll := nw.Layers[len(nw.Layers)-1]
	qlln := len(ll.Neurons)
	for llni := 0; llni < qlln; llni++ { // For every neuron in the last layer...
		lln := ll.Neurons[llni]

		lln.dLossDValue = 2 * (lln.Value - truth[llni])                   // d(MSE)
		lln.dLossDBias = lln.dLossDValue * lln.dValueDNet * lln.dNetDBias // dNetDBias is always 1, so not really needed, but included for calculus reasons.

		qc := len(lln.Connections)
		for ci := 0; ci < qc; ci++ { // For every connection from this layer to its previous layer's neurons...
			c := lln.Connections[ci]
			c.dLossDWeight = lln.dLossDValue * lln.dValueDNet * c.dNetDWeight
		}
	}

	ql := len(nw.Layers)
	fli := 0
	for li := ql - 2; li >= fli; li-- { // For every layer except the last, starting from the second to last...
		l := nw.Layers[li]
		nli := li + 1
		nl := nw.Layers[nli]

		qn := len(l.Neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := l.Neurons[ni]

			qnln := len(nl.Neurons)
			for nlni := 0; nlni < qnln; nlni++ { // For every neuron in the next layer...
				nln := nl.Neurons[nlni]

				n.dLossDValue += nln.dLossDValue * nln.dValueDNet * nln.Connections[ni].dNetDPrevValue
			}

			n.dLossDBias = n.dLossDValue * n.dValueDNet * n.dNetDBias // dNetDBias is always 1, so not really needed, but included for calculus reasons.

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ { // For every connection from this neuron to its previous layer's neurons...
				c := n.Connections[ci]
				c.dLossDWeight = n.dLossDValue * n.dValueDNet * c.dNetDWeight
			}
		}
	}

	return nw
}

func (nw *Network) recordNudges() (this *Network) {
	ql := len(nw.Layers)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.Layers[li]

		l.recordNudges()
	}

	return nw
}

func (nw *Network) averageNudges() (this *Network) {
	ql := len(nw.Layers)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.Layers[li]

		l.averageNudges()
	}

	return nw
}

func (nw *Network) adjustWeights(learningRate float64) (this *Network) {
	ql := len(nw.Layers)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.Layers[li]

		l.adjustWeights(learningRate)
	}

	return nw
}

// TODO: Unexport
// CalculateLoss determines how far off the network's current output state is from the input vector, `truth`.
// Loss is stored on the network itself.
func (nw *Network) CalculateLoss(truth []float64) (this *Network) {
	ll := nw.Layers[len(nw.Layers)-1]
	qt, qn := len(truth), len(ll.Neurons)

	if qt != qn {
		panic("Can't calculate loss, truth and output layer are of different lengths!")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in the last layer...
		n := ll.Neurons[ni]
		nw.Loss += math.Pow(n.Value-truth[ni], 2) // MSE
	} // C_0

	return nw
}

func (nw *Network) GetResults() (results string) {
	lli := len(nw.Layers) - 1
	ll := nw.Layers[lli]

	result := "Not Set"
	maxVal := -float64(math.MaxInt32)

	qlln := len(ll.Neurons)
	for ni := 0; ni < qlln; ni++ {
		n := ll.Neurons[ni]

		results = fmt.Sprintf("%v%v: %v\n", results, ni, n.Value)

		if n.Value > maxVal {
			maxVal = n.Value
			result = fmt.Sprintf("%v", ni)
		}
	}

	results = fmt.Sprintf("%vResult: %v\n", results, result)

	return results
}
