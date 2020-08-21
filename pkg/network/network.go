package network

import (
	"errors"
	"fmt"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/TheDemx27/calculus"
	"math"
)

type Network []Layer

// Spec defines the details for the construction of a Network.
// - NeuronMap is an []int which is intended to detail the number of neurons in each Layer. For example if index 3
//   contains the value 5, that would mean that the third Layer of the network should contain 5 neurons.
// - InputLabels defines the labels for each of the input neurons. len(InputLabels) must equal NeuronMap[0].
// - OutputLabels defines the labels for each of the output neurons. len(OutputLabels) must equal
//     NeuronMap[len(NeuronMap)-1]
// - ActivationFunctionName is a name corresponding to an ActivationFunction found in the activationfunction package.
//     All neurons created for this network will use this acitvation function
type Spec struct {
	NeuronMap              []int
	InputLabels            []string
	OutputLabels           []string
	ActivationFunctionName activationfunction.Name
}

// From creates a new Network from the construction details in spec.
func From(spec Spec) (Network, error) {
	nw := Network{}

	if spec.ActivationFunctionName == "" {
		return nil, errors.New("must provide an activation function name")
	}

	if spec.NeuronMap == nil {
		return nil, errors.New("must provide a neuron map")
	}
	for li := range spec.NeuronMap {
		qn := spec.NeuronMap[li]

		if li == 0 {
			l, err := NewLayer(qn, nil, spec.ActivationFunctionName)
			if err != nil {
				return nil, err
			}
			nw = append(nw, l)
			continue
		}
		pl := nw[li-1]
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
		fmt.Println("warning: you did not provide any output labels, this is likely unintended")
		spec.OutputLabels = make([]string, spec.NeuronMap[len(spec.NeuronMap)-1])
	}
	err = nw.LastLayer().SetNeuronLabels(spec.OutputLabels)
	if err != nil {
		return nil, err
	}

	return nw, nil
}

func MustFrom(spec Spec) Network {
	nw, err := From(spec)
	if err != nil {
		panic(err)
	}
	return nw
}

// Reconnect connects all the neurons in nw to all the neurons in their previous layers using brand new connections.
// This function will scan over the entire network and recreate all connections between contiguous layers.
func (nw Network) Reconnect() {
	for li := len(nw) - 1; li > 0; li-- {
		pl := nw[li-1]
		nw[li].ConnectTo(pl)
	}
}

// ReconnectWith connects all the neurons in nw to all the neurons in their previous layers using the provided
// connections. cs is an LxMxN slice of connections where L is the number of layers in n, M is the number of neurons in
// l, and n is the number of neurons in pl. If this is not honored, an error will be returned.
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

// ReconnectNeurons connects all neurons in nw to all neurons in their previous layers using the existing connections.
// It only updates what each neurons Connection.To points to, all other values are preserved.
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

func (nw Network) Predict(input []float64) (string, error) {
	if len(input) != len(nw.FirstLayer()) {
		return "", fmt.Errorf("invalid number of values provided (%v), does no match number of neurons in Layer (%v)", len(input), len(nw.FirstLayer()))
	}

	nw.ResetForPass(true)

	err := nw.ForwardPass(input)
	if err != nil {
		return "", err
	}

	return nw.HighestConfidenceNeuron().label, nil
}

func (nw Network) MustPredict(input []float64) string {
	prediction, err := nw.Predict(input)
	if err != nil {
		panic(err)
	}
	return prediction
}

func (nw Network) ForwardPass(input []float64) error {
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

func (nw Network) MustForwardPass(input []float64) {
	err := nw.ForwardPass(input)
	if err != nil {
		panic(err)
	}
}

// TODO(justin): Break up
func (nw Network) BackwardPass(truth []float64) error {
	ll := nw.LastLayer()

	if len(truth) != len(ll) {
		return fmt.Errorf("cannot perform backwards pass: truth data length (%v) is not of same length as last layer of neurons (%v)", len(truth), len(ll))
	}

	for ni := range ll {
		ll[ni].dLossDValue = 2 * (ll[ni].value - truth[ni])
		ll[ni].dLossDBias = ll[ni].dLossDValue * ll[ni].dValueDNet * ll[ni].dNetDBias
		for ci := range ll[ni].Connections { // For every Connection from this Layer to its previous Layer's neurons...
			ll[ni].Connections[ci].dLossDWeight = ll[ni].dLossDValue * ll[ni].dValueDNet * ll[ni].Connections[ci].dNetDWeight
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

	return nil
}

func (nw Network) MustBackwardPass(truth []float64) {
	err := nw.BackwardPass(truth)
	if err != nil {
		panic(err)
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

func (nw Network) MustCalculateLoss(truth []float64) float64 {
	loss, err := nw.CalculateLoss(truth)
	if err != nil {
		panic(err)
	}
	return loss
}

func (nw Network) ResetForPass(andBatch bool) {
	for li := range nw {
		nw[li].resetForPass(andBatch)
	}
}

func (nw Network) RecordNudges() {
	for li := range nw {
		nw[li].recordNudges()
	}
}

func (nw Network) AdjustWeights(learningRate float64) {
	for li := range nw {
		nw[li].adjustWeights(learningRate)
	}
}
