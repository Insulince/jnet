package jnet

import (
	"errors"
	"fmt"
	"github.com/TheDemx27/calculus"
	"math"
	"math/rand"
	"time"
)

type Network struct {
	layers   []*layer
	Metadata metadata
}

type metadata struct {
	timestamp         string
	PredictionHistory []predictionHistory
	TrainingHistory   []trainingHistory
}

// TODO: Add loss statistics and actual iterations to this struct
type trainingHistory struct {
	*TrainingConfiguration
	DataSetSize int
	Start       string
	Finish      string
}

// TODO: Add confidences to this struct
type predictionHistory struct {
	Prediction string
	Timestamp  string
	Input      []float64
	Output     []float64
}

func NewNetwork(nm []int, il []string, ol []string) (nnw *Network, err error) {
	nnw = &Network{}

	nnw.Metadata.timestamp = time.Now().Format(time.RFC3339)

	qqn := len(nm)
	for nmi := 0; nmi < qqn; nmi++ { // For every quantity of neurons in the neuron map...
		qn := nm[nmi]
		ql := nmi

		if ql == 0 {
			nnw.layers = append(nnw.layers, newLayer(qn, nil))
			continue
		}
		pl := nnw.layers[ql-1]
		nnw.layers = append(nnw.layers, newLayer(qn, pl))
	}

	ql := len(nnw.layers)
	fli := 0
	err = nnw.layers[fli].setNeuronLabels(il)
	if err != nil {
		return nil, err
	}
	lli := ql - 1
	err = nnw.layers[lli].setNeuronLabels(ol)
	if err != nil {
		return nil, err
	}

	return nnw, nil
}

func (nw *Network) resetForPass() {
	ql := len(nw.layers)
	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		l.resetForPass()
	}
}

func (nw *Network) resetForMiniBatch() {
	nw.resetForPass()

	ql := len(nw.layers)
	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		l.resetForMiniBatch()
	}
}

func (nw *Network) Predict(input []float64) (prediction string, err error) {
	defer func() {
		if err == nil {
			nw.recordPrediction(input, prediction)
		}
	}()

	nw.resetForPass()
	err = nw.forwardPass(input)
	if err != nil {
		return "", err
	}

	return nw.getHighestConfidenceNeuron().label, nil
}

func (nw *Network) recordPrediction(input []float64, prediction string) {
	var output []float64
	ll := nw.layers[len(nw.layers)-1]
	qnll := len(ll.neurons)
	for llni := 0; llni < qnll; llni++ { // For every neuron in the last layer...
		lln := ll.neurons[llni]
		output = append(output, lln.value)
	}
	nw.Metadata.PredictionHistory = append(nw.Metadata.PredictionHistory, predictionHistory{
		Input:      input,
		Output:     output,
		Prediction: prediction,
		Timestamp:  time.Now().Format(time.RFC3339),
	})
}

func (nw *Network) forwardPass(input []float64) (err error) {
	fl := nw.layers[0]
	err = fl.setNeuronValues(input)
	if err != nil {
		return err
	}

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

			net := n.wSum + n.bias
			n.value = sigmoid(net)
			n.dValueDNet = calculus.Diff(sigmoid, net)
			n.dNetDBias = 1.0

			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.connections[ci]

				c.dNetDWeight = c.left.value
				c.dNetDPrevValue = c.weight
			}
		}
	}

	return nil
}

func (nw *Network) Train(td TrainingData, tc TrainingConfiguration) (err error) {
	defer func(s string) {
		if err == nil {
			nw.recordTrainingSession(td, tc, s, time.Now().Format(time.RFC3339))
		}
	}(time.Now().Format(time.RFC3339))

	fmt.Println("Starting training process...")

	totalLoss, averageLoss, minMiniBatchLoss, maxMiniBatchLoss := 0.0, 0.0, float64(math.MaxInt32), float64(-math.MaxInt32)

	for ti := 0; ti < tc.Iterations; ti++ { // For every desired training iteration...
		miniBatch, err := td.miniBatch(tc.MiniBatchSize)
		if err != nil {
			return err
		}

		totalMiniBatchLoss := 0.0

		nw.resetForMiniBatch()
		for mbi := 0; mbi < tc.MiniBatchSize; mbi++ { // For every desired item to be in the minibatch...
			td := &miniBatch[mbi]

			nw.resetForPass()
			err = nw.forwardPass(td.Data)
			if err != nil {
				return err
			}
			loss, err := nw.calculateLoss(td.Truth)
			if err != nil {
				return err
			}
			totalMiniBatchLoss += loss
			nw.backwardPass(td.Truth)
			nw.recordNudges()
		}

		miniBatchLoss := totalMiniBatchLoss / float64(tc.MiniBatchSize) // Get the average loss across the whole mini batch.
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
			fmt.Printf(" | %5f %5f %5f - %v\n", averageLoss, minMiniBatchLoss, maxMiniBatchLoss, ti)
		}

		if averageLoss < tc.AverageLossCutoff {
			fmt.Printf("\nReached average loss cutoff limit, ending training process...\n")
			break
		}

		nw.calculateAverageNudges()
		nw.adjustWeights(tc.LearningRate)
	}

	fmt.Println("Training process ended.")

	return nil
}

func (nw *Network) recordTrainingSession(td TrainingData, tc TrainingConfiguration, start string, finish string) {
	nw.Metadata.TrainingHistory = append(nw.Metadata.TrainingHistory, trainingHistory{
		TrainingConfiguration: &tc,
		DataSetSize:           len(td),
		Start:                 start,
		Finish:                finish,
	})
}

func (nw *Network) backwardPass(truth []float64) {
	ll := nw.layers[len(nw.layers)-1]
	qlln := len(ll.neurons)
	for llni := 0; llni < qlln; llni++ { // For every neuron in the last layer...
		lln := ll.neurons[llni]

		lln.dLossDValue = 2 * (lln.value - truth[llni])
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

func (nw *Network) calculateLoss(truth []float64) (loss float64, err error) {
	ll := nw.layers[len(nw.layers)-1]
	qt, qn := len(truth), len(ll.neurons)

	if qt != qn {
		return math.NaN(), errors.New("can't calculate loss, truth and output layer are of different lengths")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in the last layer...
		n := ll.neurons[ni]
		loss += math.Pow(n.value-truth[ni], 2)
	}

	return loss, nil
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

func (nw *Network) Mutate() {
	nw.layers[rand.Intn(len(nw.layers))].mutate()
}
