package network

import (
	"errors"
	"fmt"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/Insulince/jnet/pkg/training"
	"github.com/TheDemx27/calculus"
	"math"
)

type Network []Layer

// Spec defines the details for construction a Network.
// - NeuronMap is an []int which is intended to detail the number of neurons in each Layer. For example if index 3
//   contains the value 5, that would mean that the third Layer of the network should contain 5 neurons.
// - InputLabels defines the labels for each of the input neurons. len(InputLabels) must equal NeuronMap[0].
// - OutputLabels defines the labels for each of the output neurons. len(OutputLabels) must equal
//    NeuronMap[len(NeuronMap)-1]
type Spec struct {
	NeuronMap    []int
	InputLabels  []string
	OutputLabels []string
}

// TODO(justin): Make simple constructor and fine grained constructor
type Constructor interface {
	Construct()
}

// New creates a new Network from the construction details in spec.
func New(spec Spec) (Network, error) {
	nw := Network{}

	if spec.NeuronMap == nil {
		return nil, errors.New("must provide a neuron map")
	}
	for li := range spec.NeuronMap {
		qn := spec.NeuronMap[li]

		if li == 0 {
			nw = append(nw, NewLayer(qn, nil))
			continue
		}
		pl := nw[li-1]
		nw = append(nw, NewLayer(qn, pl))
	}

	if spec.InputLabels == nil {
		spec.InputLabels = make([]string, spec.NeuronMap[0])
	}
	err := nw.FirstLayer().SetNeuronLabels(spec.InputLabels)
	if err != nil {
		return nil, err
	}

	if spec.OutputLabels == nil {
		return nil, errors.New("must provide output labels") // TODO(justin): Make optional?
	}
	err = nw.LastLayer().SetNeuronLabels(spec.OutputLabels)
	if err != nil {
		return nil, err
	}

	return nw, nil
}

func MustNew(spec Spec) Network {
	nw, err := New(spec)
	if err != nil {
		panic(err)
	}
	return nw
}

func (nw Network) Predict(input []float64) (string, error) {
	if len(input) != len(nw.FirstLayer()) {
		return "", fmt.Errorf("invalid number of values provided (%v), does no match number of neurons in Layer (%v)", len(input), len(nw.FirstLayer()))
	}

	nw.resetForPass()

	err := nw.forwardPass(input, activationfunction.Sigmoid) // TODO Defaults to Sigmoid, should this be changed?
	if err != nil {
		return "", err
	}

	return nw.HighestConfidenceNeuron().label, nil
}

func (nw Network) forwardPass(input []float64, activationFunction activationfunction.ActivationFunction) error {
	err := nw.FirstLayer().SetNeuronValues(input)
	if err != nil {
		return err
	}

	for li := 1; li < len(nw); li++ { // For every layer EXCEPT THE FIRST, starting from the SECOND...
		l := nw[li]
		for ni := range l {
			n := l[ni]
			for ci := range n.Connections {
				c := n.Connections[ci]
				n.wSum += c.To.value * c.weight
			}

			net := n.wSum + n.bias
			n.value = activationFunction(net)
			n.dValueDNet = calculus.Diff((func(float64) float64)(activationFunction), net)
			n.dNetDBias = 1.0

			for ci := range n.Connections {
				c := n.Connections[ci]
				c.dNetDWeight = c.To.value
				c.dNetDPrevValue = c.weight
			}
		}
	}

	return nil
}

// TODO(justin): Add a return type that is a cancel function
func (nw Network) Train(td training.Data, tc training.Configuration) error {
	// TODO(justin): Add validation that TC is valid (must contain an activation function. or just set a default)
	// TODO(justin): Make use of the timeout in the training configuration.

	fmt.Println("Starting training process...")

	totalLoss, averageLoss, minMiniBatchLoss, maxMiniBatchLoss := 0.0, 0.0, float64(math.MaxInt32), float64(-math.MaxInt32)

	for ti := 0; ti < tc.Iterations; ti++ { // For every desired training iteration...
		miniBatch, err := td.MiniBatch(tc.MiniBatchSize)
		if err != nil {
			return err
		}

		totalMiniBatchLoss := 0.0

		nw.resetForMiniBatch()
		for _, td := range miniBatch {
			nw.resetForPass()

			err := nw.forwardPass(td.Data, tc.ActivationFunction)
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
		averageLoss = totalLoss / float64(ti) // TODO divide by zero????

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

		nw.adjustWeights(tc.LearningRate)
	}

	fmt.Println("Training process ended.")

	return nil
}

// TODO(justin): Break up
// TODO(justin): Either make this also return an error like forwardPass or make forwardPass swallow the error for consistency.
func (nw Network) backwardPass(truth []float64) {
	ll := nw.LastLayer()
	for i := range ll {
		ll[i].dLossDValue = 2 * (ll[i].value - truth[i])
		ll[i].dLossDBias = ll[i].dLossDValue * ll[i].dValueDNet * ll[i].dNetDBias
		for ci := range ll[i].Connections { // For every Connection from this Layer to its previous Layer's neurons...
			ll[i].Connections[ci].dLossDWeight = ll[i].dLossDValue * ll[i].dValueDNet * ll[i].Connections[ci].dNetDWeight
		}
	}

	for li := len(nw) - 2; li >= 0; li-- { // For every Layer except the last, starting from the second to last...
		l := nw[li]
		nli := li + 1
		nl := nw[nli]

		for ni := range l { // For every neuron in this layer...
			for nni := range nl { // For every neuron in the next layer...
				l[ni].dLossDValue += nl[nni].dLossDValue * nl[nni].dValueDNet * nl[nni].Connections[ni].dNetDPrevValue
			}
			l[ni].dLossDBias = l[ni].dLossDValue * l[ni].dValueDNet * l[ni].dNetDBias

			for ci := range l[ni].Connections { // For every Connection from this Neuron to its previous Layer's neurons...
				l[ni].Connections[ci].dLossDWeight = l[ni].dLossDValue * l[ni].dValueDNet * l[ni].Connections[ci].dNetDWeight
			}
		}
	}
}

func (nw Network) HighestConfidenceNeuron() *Neuron {
	ll := nw.LastLayer()
	var hcn = ll[0]
	for ni := range nw.LastLayer() {
		if ll[ni].value > hcn.value {
			hcn = ll[ni]
		}
	}
	return hcn
}

func (nw Network) calculateLoss(truth []float64) (float64, error) {
	qt, qn := len(truth), len(nw.LastLayer())
	if qt != qn {
		return 0, fmt.Errorf("can't calculate loss, length of truth (%v) and length of output Layer (%v) do not match", qt, qn)
	}

	var loss float64
	for ni, n := range nw.LastLayer() {
		loss += math.Pow(n.value-truth[ni], 2)
	}
	return loss, nil
}

func (nw Network) mustCalculateLoss(truth []float64) float64 {
	loss, err := nw.calculateLoss(truth)
	if err != nil {
		panic(err)
	}
	return loss
}

func (nw Network) resetForPass() {
	for li := range nw {
		nw[li].resetForPass()
	}
}

func (nw Network) resetForMiniBatch() {
	for li := range nw {
		nw[li].resetForMiniBatch()
	}
}

func (nw Network) recordNudges() {
	for li := range nw {
		nw[li].recordNudges()
	}
}

func (nw Network) adjustWeights(learningRate float64) {
	for li := range nw {
		nw[li].adjustWeights(learningRate)
	}
}
