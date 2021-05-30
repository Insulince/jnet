package network

import (
	"errors"
	"fmt"
	"math"

	"github.com/TheDemx27/calculus"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
)

// Network is the top level type for interacting with the neural networks this
// package provides.
type Network []Layer

// Spec defines the details for the construction of a Network.
type Spec struct {
	// NeuronMap is intended to detail the number of layers in this network as
	// well as the number of neurons in each Layer. For example if index 3
	// contains the value 5, that would mean that the third Layer of the network
	// should contain 5 neurons.
	NeuronMap []int
	// InputLabels defines the labels for each of the input neurons.
	// len(InputLabels) must equal NeuronMap[0].
	InputLabels []string
	// OutputLabels defines the labels for each of the output neurons.
	// len(OutputLabels) must equal NeuronMap[len(NeuronMap)-1].
	OutputLabels []string
	// ActivationFunctionName is a name corresponding to an ActivationFunction
	// found in the activationfunction package. All neurons created for this
	// network will use this activation function.
	ActivationFunctionName activationfunction.Name
}

// From creates a new fully-connected Network from the construction details in
// spec and returns an error if spec is invalid.
func From(spec Spec) (Network, error) {
	nw := Network{}

	if spec.ActivationFunctionName == "" {
		return nil, errors.New("must provide an activation function name")
	}

	if spec.NeuronMap == nil {
		return nil, errors.New("must provide a neuron map")
	}
	if len(spec.NeuronMap) < 2 {
		return nil, errors.New("must provide a neuron map with at least 2 layers (for input and output layer)")
	}
	for li := range spec.NeuronMap {
		qn := spec.NeuronMap[li]

		if qn < 1 {
			return nil, errors.New("layer in neuron map must contain at least one neuron")
		}

		var pl Layer
		if li > 0 {
			pl = nw[li-1]
		}
		l, err := NewLayer(qn, pl, spec.ActivationFunctionName)
		if err != nil {
			return nil, err
		}
		nw = append(nw, l)
	}

	if spec.InputLabels == nil {
		spec.InputLabels = make([]string, spec.NeuronMap[0])
	}
	err := nw.FirstLayer().SetNeuronLabels(spec.InputLabels)
	if err != nil {
		return nil, err
	}

	if spec.OutputLabels == nil {
		fmt.Println("warning: you did not provide any output labels")
		spec.OutputLabels = make([]string, spec.NeuronMap[len(spec.NeuronMap)-1])
	}
	err = nw.LastLayer().SetNeuronLabels(spec.OutputLabels)
	if err != nil {
		return nil, err
	}

	return nw, nil
}

// MustFrom calls From but panics if an error is encountered.
func MustFrom(spec Spec) Network {
	nw, err := From(spec)
	if err != nil {
		panic(err)
	}
	return nw
}

// Reconnect connects all the neurons in nw to all the neurons in their previous
// layers using brand new connections. This function will scan over the entire
// network and recreate all connections between contiguous layers.
func (nw Network) Reconnect() {
	for li := len(nw) - 1; li > 0; li-- {
		pl := nw[li-1]
		nw[li].ConnectTo(pl)
	}
}

// ReconnectWith connects all the neurons in nw to all neurons in their previous
// layers using the provided connections. cs is an LxMxN slice of connections
// where L is the number of layers in nw, M is the number of neurons in l, and n
// is the number of neurons in pl. If this is not honored, an error will be
// returned.
func (nw Network) ReconnectWith(cs [][][]*Connection) error {
	if len(cs) != len(nw) {
		return fmt.Errorf("cannot reconnect network using provided connections: number of provided sets of sets of connections (%v) does not match number of layers in network (%v)", len(cs), len(nw))
	}

	for li := len(nw) - 1; li > 0; li-- {
		pl := nw[li-1]
		err := nw[li].ConnectWith(pl, cs[li])
		if err != nil {
			return err
		}
	}

	return nil
}

// ReconnectNeurons connects all neurons in nw to all neurons in their previous
// layers using the existing connections. It only updates what each neurons
// Connection.To points to, all other values are preserved.
func (nw Network) ReconnectNeurons() error {
	for li := len(nw) - 1; li > 0; li-- {
		pl := nw[li-1]
		err := nw[li].ConnectNeurons(pl)
		if err != nil {
			return err
		}
	}

	return nil
}

// IsFullyConnected reports whether all neurons in the network are connected to
// another neuron by checking that the neuron has the same number of connections
// as there are neurons in the previous layer and that each of those connections
// has its To field populated with a non-nil neuron.
func (nw Network) IsFullyConnected() bool {
	for li := 1; li < len(nw); li++ {
		l := nw[li]
		pl := nw[li-1]

		for ni := 0; ni < len(l); ni++ {
			n := l[ni]

			if len(n.Connections) != len(pl) {
				return false
			}

			for ci := 0; ci < len(n.Connections); ci++ {
				c := n.Connections[ci]

				if c.To == nil {
					return false
				}
			}
		}
	}

	return true
}

// Predict will execute a ForwardPass on nw using input as its input, then
// returns the label and value of the neuron with the highest confidence. If you
// wish to see all output neurons instead of just the neuron with highest
// confidence then use LastLayer to inspect them all.
//
// If len(input) != len(nw.FirstLayer()) then an error will be returned.
func (nw Network) Predict(input []float64) (string, float64, error) {
	if len(input) != len(nw.FirstLayer()) {
		return "", 0, fmt.Errorf("invalid number of values provided (%v), does no match number of neurons in Layer (%v)", len(input), len(nw.FirstLayer()))
	}

	nw.ResetFromBatch()

	err := nw.ForwardPass(input)
	if err != nil {
		return "", 0, err
	}

	hcn := nw.HighestConfidenceNeuron()

	return hcn.label, hcn.value, nil
}

// MustPredict calls Predict but panics if an error is encountered.
func (nw Network) MustPredict(input []float64) (string, float64) {
	prediction, value, err := nw.Predict(input)
	if err != nil {
		panic(err)
	}
	return prediction, value
}

// ForwardPass executes a forward pass on nw with input fed into nw's input
// layer index-wise.
//
// nw is mutated during this process to track the weighted sum of all inputs as
// its fed through the network as well as the calculus required to do back
// propagation and adjust the weights accordingly.
//
// if len(input) != len(nw.FirstLayer()) then an error will be returned.
func (nw Network) ForwardPass(input []float64) error {
	err := nw.FirstLayer().SetNeuronValues(input)
	if err != nil {
		return err
	}

	// For every layer EXCEPT THE FIRST, starting from the SECOND...
	for li := 1; li < len(nw); li++ {
		l := nw[li]
		for ni := range l {
			n := l[ni]
			for ci := range n.Connections {
				c := n.Connections[ci]
				n.wSum += c.To.value * c.weight
				c.dNetDWeight = c.To.value
				c.dNetDPrevValue = c.weight
			}

			net := n.wSum + n.bias
			n.value = n.activationFunction(net)
			n.dValueDNet = calculus.Diff((func(float64) float64)(n.activationFunction), net)
			n.dNetDBias = 1.0
		}
	}

	return nil
}

// MustForwardPass calls ForwardPass but panics if an error is encountered.
func (nw Network) MustForwardPass(input []float64) {
	err := nw.ForwardPass(input)
	if err != nil {
		panic(err)
	}
}

// BackwardPass executes a backward pass on nw with truth compared to nw's
// output layer index-wise in order to calculate back propagation of the loss
// value across nw's constituent parts.
//
// nw is mutated during this process to track the weighted sum of all inputs as
// its fed through the network as well as the calculus required to do back
// propagation and adjust the weights accordingly.
//
// if len(truth) != len(nw.LastLayer()) then an error will be returned.
// TODO(justin): Break up
func (nw Network) BackwardPass(truth []float64) error {
	ll := nw.LastLayer()

	if len(truth) != len(ll) {
		return fmt.Errorf("cannot perform backwards pass: truth data length (%v) is not of same length as last layer of neurons (%v)", len(truth), len(ll))
	}

	for ni := range ll {
		ll[ni].dLossDValue = 2 * (ll[ni].value - truth[ni])
		ll[ni].dLossDBias = ll[ni].dLossDValue * ll[ni].dValueDNet * ll[ni].dNetDBias
		// For every Connection from this Layer to its previous Layer's neurons...
		for ci := range ll[ni].Connections {
			ll[ni].Connections[ci].dLossDWeight = ll[ni].dLossDValue * ll[ni].dValueDNet * ll[ni].Connections[ci].dNetDWeight
		}
	}

	// For every Layer except the last, starting from the second to last...
	for li := len(nw) - 2; li >= 0; li-- {
		l := nw[li]
		nli := li + 1
		nl := nw[nli]

		for ni := range l { // For every neuron in this layer...
			for nni := range nl { // For every neuron in the next layer...
				l[ni].dLossDValue += nl[nni].dLossDValue * nl[nni].dValueDNet * nl[nni].Connections[ni].dNetDPrevValue
			}
			l[ni].dLossDBias = l[ni].dLossDValue * l[ni].dValueDNet * l[ni].dNetDBias

			// For every Connection from this Neuron to its previous Layer's
			// neurons...
			for ci := range l[ni].Connections {
				l[ni].Connections[ci].dLossDWeight = l[ni].dLossDValue * l[ni].dValueDNet * l[ni].Connections[ci].dNetDWeight
			}
		}
	}

	return nil
}

// MustBackwardPass calls BackwardPass but panics if an error is encountered.
func (nw Network) MustBackwardPass(truth []float64) {
	err := nw.BackwardPass(truth)
	if err != nil {
		panic(err)
	}
}

// HighestConfidenceNeuron iterates over every neuron in nw's last layer and
// returns the one with the highest value (confidence).
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

// CalculateLoss returns the loss value of the current state of nw's output
// layer as compared with the values in truth. This should be used after running
// ForwardPass to see the true value of loss for a given input.
//
// If len(truth) != len(nw.LastLayer()) then an error will be returned.
func (nw Network) CalculateLoss(truth []float64) (float64, error) {
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

// MustCalculateLoss calls CalculateLoss but panics if an error is encountered.
func (nw Network) MustCalculateLoss(truth []float64) float64 {
	loss, err := nw.CalculateLoss(truth)
	if err != nil {
		panic(err)
	}
	return loss
}

// ResetFromBatch will reset the entire network from the perspective of having
// just ran an entire mini batch. This differs from ResetFromPass in that it
// will reset all the nudges recorded on the bias and weights of the network,
// not just the calculus used for back propagation.
//
// Call this function after executing a forward and backward pass for an entire
// mini batch.
func (nw Network) ResetFromBatch() {
	for li := range nw {
		nw[li].resetFromBatch()
	}
}

// ResetFromPass will reset the entire network from the perspective of having
// just ran a forward and backward pass. This differs from ResetFromBatch in
// that it will only reset the calculus used for back propagation, not the
// nudges recorded on the bias and weights of each neuron.
//
// Call this function after executing a forward and backward pass on the network
// once.
func (nw Network) ResetFromPass() {
	for li := range nw {
		nw[li].resetFromPass()
	}
}

// RecordNudges will record the values calculated for gradient descent in nw's
// relevant places so that after a Forward and Backward pass the direction of
// progress towards minimizing the loss function can be recorded.
//
// This should be called for each time a forward and backward pass is executed
// in a mini batch.
//
// TODO(justin): I feel like RecordNudges should be part of or called in
//  BackwardPass since its only ever used directly after calling that and
//  wouldn't have much value to an end user.
func (nw Network) RecordNudges() {
	for li := range nw {
		nw[li].recordNudges()
	}
}

// AdjustWeights will nudge all the weights and biases across the entire network
// in the direction of progress towards minimizing the loss function by taking
// the average value across that neuron or connections nudge-values and pushing
// the weight in that direction scaled against learningRate.
//
// This should be called after executing a forward and backward pass for an
// entire mini batch.
func (nw Network) AdjustWeights(learningRate float64) {
	for li := range nw {
		nw[li].adjustWeights(learningRate)
	}
}
