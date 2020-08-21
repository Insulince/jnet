package network

import (
	"fmt"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
)

//////////////////////////////////////// Network ////////////////////////////////////////

func (nw Network) Equals(nw2 Network) error {
	if len(nw) != len(nw2) {
		return fmt.Errorf("networks are different lengths, %v != %v", len(nw), len(nw2))
	}

	for li := range nw {
		if err := nw[li].Equals(nw2[li]); err != nil {
			return err
		}
	}

	return nil
}

func (nw Network) NumLayers() int {
	return len(nw)
}

func (nw Network) FirstLayer() Layer {
	return nw[0]
}

func (nw Network) LastLayer() Layer {
	return nw[len(nw)-1]
}

func (nw Network) GetLayer(i int) (Layer, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get layer at index < 0 (requested %v)", i)
	}
	if i >= len(nw) {
		return nil, fmt.Errorf("cannot get layer at index > size of network, %v (requested %v)", len(nw), i)
	}
	return nw[i], nil
}

func (nw Network) MustGetLayer(i int) Layer {
	l, err := nw.GetLayer(i)
	if err != nil {
		panic(err)
	}
	return l
}

// GetLayers returns the subset of layers from i to j, i inclusive and j exclusive.
func (nw Network) GetLayers(i, j int) ([]Layer, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(nw) {
		return nil, fmt.Errorf("cannot get subset ending at > size of network (requested j: %v)", j)
	}
	return nw[i:j], nil
}

func (nw Network) MustGetLayers(i, j int) []Layer {
	ls, err := nw.GetLayers(i, j)
	if err != nil {
		panic(err)
	}
	return ls
}

func (nw Network) SetLayer(i int, l Layer) error {
	if i < 0 {
		return fmt.Errorf("cannot set layer at index < 0 (requested %v)", i)
	}
	if i >= len(nw) {
		return fmt.Errorf("cannot set layer at index > size of network, %v (requested %v)", len(nw), i)
	}
	nw[i] = l
	if i > 0 {
		l.ConnectTo(nw[i-1])
	}
	return nil
}

func (nw Network) MustSetLayer(i int, l Layer) {
	err := nw.SetLayer(i, l)
	if err != nil {
		panic(err)
	}
}

// SetLayers sets the subset of layers from i to j, i inclusive and j exclusive, to ls.
func (nw Network) SetLayers(i, j int, ls []Layer) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(nw) {
		return fmt.Errorf("cannot set subset ending at > size of network (requested j: %v)", j)
	}
	q := j - i
	if len(ls) != q {
		return fmt.Errorf("cannot set a subset of layers to a set of layers of different length. target subset length: %v, provided set length: %v", q, len(ls))
	}
	for k := 0; k < q; k++ {
		nw[k+i] = ls[k]
		if k+i > 0 {
			ls[k].ConnectTo(nw[k+i-1])
		}
	}
	return nil
}

func (nw Network) MustSetLayers(i, j int, ls []Layer) {
	err := nw.SetLayers(i, j, ls)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SwapOutLayer(i int, l Layer) (Layer, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap layer at index < 0 (requested %v)", i)
	}
	if i >= len(nw) {
		return nil, fmt.Errorf("cannot swap layer at index > size of network, %v (requested %v)", len(nw), i)
	}
	ol := nw.MustGetLayer(i)
	nw.MustSetLayer(i, l)
	return ol, nil
}

func (nw Network) MustSwapOutLayer(i int, l Layer) Layer {
	ol, err := nw.SwapOutLayer(i, l)
	if err != nil {
		panic(err)
	}
	return ol
}

func (nw Network) SwapOutLayers(i, j int, ls []Layer) ([]Layer, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(nw) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of network (requested j: %v)", j)
	}
	q := j - i
	if len(ls) != q {
		return nil, fmt.Errorf("cannot swap a subset of layers with a set of layers of different length. target subset length: %v, provided set length: %v", q, len(ls))
	}
	ols := nw.MustGetLayers(i, j)
	nw.MustSetLayers(i, j, ls)
	return ols, nil
}

func (nw Network) MustSwapOutLayers(i, j int, ls []Layer) []Layer {
	ols, err := nw.SwapOutLayers(i, j, ls)
	if err != nil {
		panic(err)
	}
	return ols
}

func (nw Network) SetNeuronValues(values [][]float64) error {
	if len(values) != len(nw) {
		return fmt.Errorf("invalid number of sets of values provided (%v), does not match number of layers in network (%v)", len(values), len(nw))
	}

	for li := range nw {
		if len(values[li]) != len(nw[li]) {
			return fmt.Errorf("invalid number of values provided (%v) in set %v, does not match number of neurons in layer (%v)", len(values[li]), li, len(nw))
		}
	}

	for li := range nw {
		err := nw[li].SetNeuronValues(values[li])
		if err != nil {
			return err
		}
	}

	return nil
}

func (nw Network) MustSetNeuronValues(values [][]float64) {
	err := nw.SetNeuronValues(values)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SetNeuronValuesTo(value float64) {
	for li := range nw {
		nw[li].SetNeuronValuesTo(value)
	}
}

func (nw Network) SetNeuronLabels(labels [][]string) error {
	if len(labels) != len(nw) {
		return fmt.Errorf("invalid number of sets of labels provided (%v), does not match number of layers in network (%v)", len(labels), len(nw))
	}

	for li := range nw {
		if len(labels[li]) != len(nw[li]) {
			return fmt.Errorf("invalid number of labels provided (%v) in set %v, does not match number of neurons in layer (%v)", len(labels[li]), li, len(nw))
		}
	}

	for li := range nw {
		err := nw[li].SetNeuronLabels(labels[li])
		if err != nil {
			return err
		}
	}

	return nil
}

func (nw Network) MustSetNeuronLabels(labels [][]string) {
	err := nw.SetNeuronLabels(labels)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SetNeuronLabelsTo(label string) {
	for li := range nw {
		nw[li].SetNeuronLabelsTo(label)
	}
}

func (nw Network) SetInputNeuronLabels(labels []string) error {
	if len(nw.FirstLayer()) != len(labels) {
		return fmt.Errorf("invalid number of labels provided (%v), does not match number of neurons in input layer (%v)", len(labels), len(nw.FirstLayer()))
	}

	err := nw.FirstLayer().SetNeuronLabels(labels)
	if err != nil {
		return err
	}
	return nil
}

func (nw Network) MustSetInputNeuronLabels(labels []string) {
	err := nw.SetInputNeuronLabels(labels)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SetInputNeuronLabelsTo(label string) {
	nw.FirstLayer().SetNeuronLabelsTo(label)
}

func (nw Network) SetOutputNeuronLabels(labels []string) error {
	if len(nw.LastLayer()) != len(labels) {
		return fmt.Errorf("invalid number of labels provided (%v), does not match number of neurons in output layer (%v)", len(labels), len(nw.LastLayer()))
	}

	err := nw.LastLayer().SetNeuronLabels(labels)
	if err != nil {
		panic(err)
	}
	return nil
}

func (nw Network) MustSetOutputNeuronLabels(labels []string) {
	err := nw.SetOutputNeuronLabels(labels)
	if err != nil {
		panic(err)
	}

}

func (nw Network) SetOutputNeuronLabelsTo(label string) {
	nw.LastLayer().SetNeuronLabelsTo(label)
}

func (nw Network) SetNeuronBiases(biases [][]float64) error {
	if len(biases) != len(nw) {
		return fmt.Errorf("invalid number of sets of biases provided (%v), does not match number of layers in network (%v)", len(biases), len(nw))
	}

	for li := range nw {
		if len(biases[li]) != len(nw[li]) {
			return fmt.Errorf("invalid number of biases provided (%v) in set %v, does not match number of neurons in layer (%v)", len(biases[li]), li, len(nw))
		}
	}

	for li := range nw {
		err := nw[li].SetNeuronBiases(biases[li])
		if err != nil {
			return err
		}
	}

	return nil
}

func (nw Network) MustSetNeuronBiases(biases [][]float64) {
	err := nw.SetNeuronBiases(biases)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SetNeuronBiasesTo(val float64) {
	for li := range nw {
		nw[li].SetNeuronBiasesTo(val)
	}
}

func (nw Network) SetNeuronActivationFunctions(activationFunctionNames [][]activationfunction.Name) error {
	if len(activationFunctionNames) != len(nw) {
		return fmt.Errorf("invalid number of sets of activation functions provided (%v), does not match number of layers in network (%v)", len(activationFunctionNames), len(nw))
	}

	for li := range nw {
		if len(activationFunctionNames[li]) != len(nw[li]) {
			return fmt.Errorf("invalid number of activation functions provided (%v) in set %v, does not match number of neurons in layer (%v)", len(activationFunctionNames[li]), li, len(nw))
		}
	}

	for li := range nw {
		err := nw[li].SetActivationFunctions(activationFunctionNames[li])
		if err != nil {
			return err
		}
	}

	return nil
}

func (nw Network) MustSetNeuronActivationFunctions(activationFunctionNames [][]activationfunction.Name) {
	err := nw.SetNeuronActivationFunctions(activationFunctionNames)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SetNeuronActivationFunctionsTo(activationFunctionName activationfunction.Name) error {
	for li := range nw {
		err := nw[li].SetNeuronActivationFunctionsTo(activationFunctionName)
		if err != nil {
			return err
		}
	}
	return nil
}

func (nw Network) MustSetNeuronActivationFunctionsTo(activationFunctionName activationfunction.Name) {
	err := nw.SetNeuronActivationFunctionsTo(activationFunctionName)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SetConnectionWeights(weights [][][]float64) error {
	if len(weights) != len(nw) {
		return fmt.Errorf("invalid number of sets of sets of weights provided (%v), does not match number of layers in network (%v)", len(weights), len(nw))
	}

	for li := range nw {
		if len(weights[li]) != len(nw[li]) {
			return fmt.Errorf("invalid number of sets of weights provided (%v) in set %v, does not match number of neurons in layer (%v)", len(weights[li]), li, len(nw))
		}

		for ni := range nw[li] {
			if len(weights[li][ni]) != len(nw[li][ni].Connections) {
				return fmt.Errorf("invalid number of weights provided (%v) in set %v subset %v, does not match number of connections in neurons (%v)", len(weights[li][ni]), li, ni, len(nw))
			}
		}
	}

	for li := range nw {
		err := nw[li].SetConnectionWeights(weights[li])
		if err != nil {
			return err
		}
	}

	return nil
}

func (nw Network) MustSetConnectionWeights(weights [][][]float64) {
	err := nw.SetConnectionWeights(weights)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SetConnectionWeightsTo(weight float64) {
	for li := range nw {
		nw[li].SetConnectionWeightsTo(weight)
	}
}

//////////////////////////////////////// Layer ////////////////////////////////////////

func (l Layer) Equals(l2 Layer) error {
	if len(l) != len(l2) {
		return fmt.Errorf("layers are different lengths, %v != %v", len(l), len(l2))
	}

	for ni := range l {
		if err := l[ni].Equals(l2[ni]); err != nil {
			return err
		}
	}

	return nil
}

func (l Layer) NumNeurons() int {
	return len(l)
}

func (l Layer) FirstNeuron() *Neuron {
	return l[0]
}

func (l Layer) LastNeuron() *Neuron {
	return l[len(l)-1]
}

func (l Layer) GetNeuron(i int) (*Neuron, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return nil, fmt.Errorf("cannot get neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	return l[i], nil
}

func (l Layer) MustGetNeuron(i int) *Neuron {
	n, err := l.GetNeuron(i)
	if err != nil {
		panic(err)
	}
	return n
}

func (l Layer) GetNeurons(i, j int) ([]*Neuron, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return nil, fmt.Errorf("cannot get subset ending at > size of layer (requested j: %v)", j)
	}
	return l[i:j], nil
}

func (l Layer) MustGetNeurons(i, j int) []*Neuron {
	ons, err := l.GetNeurons(i, j)
	if err != nil {
		panic(err)
	}
	return ons
}

func (l Layer) SetNeuron(i int, n *Neuron, pl Layer) error {
	if i < 0 {
		return fmt.Errorf("cannot set neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return fmt.Errorf("cannot set neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	l[i] = n
	if i > 0 {
		n.ConnectTo(pl)
	}
	return nil
}

func (l Layer) MustSetNeuron(i int, n *Neuron, pl Layer) {
	err := l.SetNeuron(i, n, pl)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeurons(i, j int, ns []*Neuron, pl Layer) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return fmt.Errorf("cannot set subset ending at > size of layer (requested j: %v)", j)
	}
	q := j - i
	if len(ns) != q {
		return fmt.Errorf("cannot set a subset of neurons to a set of neurons of different length. target subset length: %v, provided set length: %v", q, len(ns))
	}
	for k := 0; k < q; k++ {
		l[k+i] = ns[k]
		if k+i > 0 {
			ns[k].ConnectTo(pl)
		}
	}
	return nil
}

func (l Layer) MustSetNeurons(i, j int, ns []*Neuron, pl Layer) {
	err := l.SetNeurons(i, j, ns, pl)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SwapOutNeuron(i int, n *Neuron, pl Layer) (*Neuron, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return nil, fmt.Errorf("cannot swap neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	on := l.MustGetNeuron(i)
	l.MustSetNeuron(i, n, pl)
	return on, nil
}

func (l Layer) MustSwapOutNeuron(i int, n *Neuron, pl Layer) *Neuron {
	on, err := l.SwapOutNeuron(i, n, pl)
	if err != nil {
		panic(err)
	}
	return on
}

func (l Layer) SwapOutNeurons(i, j int, ns []*Neuron, pl Layer) ([]*Neuron, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of layer (requested j: %v)", j)
	}
	q := j - i
	if len(ns) != q {
		return nil, fmt.Errorf("cannot swap a subset of neurons with a set of neurons of different length. target subset length: %v, provided set length: %v", q, len(ns))
	}
	ols := l.MustGetNeurons(i, j)
	l.MustSetNeurons(i, j, ns, pl)
	return ols, nil
}

func (l Layer) MustSwapOutNeurons(i, j int, ns []*Neuron, pl Layer) []*Neuron {
	ons, err := l.SwapOutNeurons(i, j, ns, pl)
	if err != nil {
		panic(err)
	}
	return ons
}

func (l Layer) SetNeuronValues(values []float64) error {
	if len(l) != len(values) {
		return fmt.Errorf("invalid number of values provided (%v), does not match number of neurons in layer (%v)", len(values), len(l))
	}

	for ni := range l {
		l[ni].SetValue(values[ni])
	}
	return nil
}

func (l Layer) MustSetNeuronValues(values []float64) {
	err := l.SetNeuronValues(values)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeuronValuesTo(value float64) {
	for ni := range l {
		l[ni].SetValue(value)
	}
}

func (l Layer) SetNeuronLabels(labels []string) error {
	if len(l) != len(labels) {
		return fmt.Errorf("invalid number of labels provided (%v), does not match number of neurons in Layer (%v)", len(labels), len(l))
	}

	for ni := range l {
		l[ni].SetLabel(labels[ni])
	}

	return nil
}

func (l Layer) MustSetNeuronLabels(labels []string) {
	err := l.SetNeuronLabels(labels)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeuronLabelsTo(label string) {
	for ni := range l {
		l[ni].SetLabel(label)
	}
}

func (l Layer) SetNeuronBiases(biases []float64) error {
	if len(l) != len(biases) {
		return fmt.Errorf("invalid number of biases provided (%v), does not match number of neurons in layer (%v)", len(biases), len(l))
	}

	for ni := range l {
		l[ni].SetBias(biases[ni])
	}
	return nil
}

func (l Layer) MustSetNeuronBiases(biases []float64) {
	err := l.SetNeuronBiases(biases)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeuronBiasesTo(value float64) {
	for ni := range l {
		l[ni].SetBias(value)
	}
}

func (l Layer) SetActivationFunctions(activationFunctionNames []activationfunction.Name) error {
	if len(l) != len(activationFunctionNames) {
		return fmt.Errorf("invalid number of activation functions provided (%v), does not match number of neurons in layer (%v)", len(activationFunctionNames), len(l))
	}

	for ni := range l {
		err := l[ni].SetActivationFunction(activationFunctionNames[ni])
		if err != nil {
			return err
		}
	}
	return nil
}

func (l Layer) MustSetActivationFunctions(activationFunctionNames []activationfunction.Name) {
	err := l.SetActivationFunctions(activationFunctionNames)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeuronActivationFunctionsTo(activationFunctionName activationfunction.Name) error {
	for ni := range l {
		err := l[ni].SetActivationFunction(activationFunctionName)
		if err != nil {
			return err
		}
	}
	return nil
}

func (l Layer) MustSetNeuronActivationFunctionsTo(activationFunctionName activationfunction.Name) {
	err := l.SetNeuronActivationFunctionsTo(activationFunctionName)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetConnectionWeights(weights [][]float64) error {
	if len(weights) != len(l) {
		return fmt.Errorf("invalid number of sets of weights provided (%v), does not match number of neurons in layer (%v)", len(weights), len(l))
	}

	for ni := range l {
		if len(weights[ni]) != len(l[ni].Connections) {
			return fmt.Errorf("invalid number of weights provided (%v) in set %v, does not match number of connections in neurons (%v)", len(weights[ni]), ni, len(l))
		}
	}

	for ni := range l {
		err := l[ni].SetConnectionWeights(weights[ni])
		if err != nil {
			return err
		}
	}

	return nil
}

func (l Layer) MustSetConnectionWeights(weights [][]float64) {
	err := l.SetConnectionWeights(weights)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetConnectionWeightsTo(weight float64) {
	for ni := range l {
		l[ni].SetConnectionWeightsTo(weight)
	}
}

//////////////////////////////////////// Neuron ////////////////////////////////////////

func (n *Neuron) Equals(n2 *Neuron) error {
	// NOTE(justin): To avoid misleading behavior, nil values are not considered equal.
	if n == nil || n2 == nil {
		return fmt.Errorf("one or both of the neurons are nil, %v || %v", n == nil, n2 == nil)
	}

	if len(n.Connections) != len(n2.Connections) {
		return fmt.Errorf("neurons do not have same number of connections, %v != %v", len(n.Connections), len(n2.Connections))
	}
	for ci := range n.Connections {
		if err := n.Connections[ci].Equals(n2.Connections[ci]); err != nil {
			return err
		}
	}

	if n.label != n2.label {
		return fmt.Errorf("neurons' labels do not match, %v != %v", n.label, n2.label)
	}
	if n.value != n2.value {
		return fmt.Errorf("neurons' values do not match, %v != %v", n.value, n2.value)
	}
	if n.wSum != n2.wSum {
		return fmt.Errorf("neurons' wSums do not match, %v != %v", n.wSum, n2.wSum)
	}
	if n.bias != n2.bias {
		return fmt.Errorf("neurons' biases do not match, %v != %v", n.bias, n2.bias)
	}
	if n.dLossDValue != n2.dLossDValue {
		return fmt.Errorf("neurons' dLossDValues do not match, %v != %v", n.dLossDValue, n2.dLossDValue)
	}
	if n.dLossDBias != n2.dLossDBias {
		return fmt.Errorf("neurons' dLossDBias do not match, %v != %v", n.dLossDBias, n2.dLossDBias)
	}
	if n.dValueDNet != n2.dValueDNet {
		return fmt.Errorf("neurons' dValueDNet do not match, %v != %v", n.dValueDNet, n2.dValueDNet)
	}
	if n.dNetDBias != n2.dNetDBias {
		return fmt.Errorf("neurons' dNetDBias do not match, %v != %v", n.dNetDBias, n2.dNetDBias)
	}

	if len(n.biasNudges) != len(n2.biasNudges) {
		return fmt.Errorf("neurons do not have same number of bias nudges, %v != %v", len(n.biasNudges), n2.biasNudges)
	}
	for bi := range n.biasNudges {
		if n.biasNudges[bi] != n2.biasNudges[bi] {
			return fmt.Errorf("neurons' biasNudges do not match, %v != %v", n.biasNudges[bi], n2.biasNudges[bi])
		}
	}

	return nil
}

func (n *Neuron) NumConnections() int {
	return len(n.Connections)
}

func (n *Neuron) FirstConnection() *Connection {
	return n.Connections[0]
}

func (n *Neuron) LastConnection() *Connection {
	return n.Connections[len(n.Connections)-1]
}

func (n *Neuron) GetConnection(i int) (*Connection, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get Connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return nil, fmt.Errorf("cannot get Connection at index > size of Neuron, %v (requested %v)", len(n.Connections), i)
	}
	return n.Connections[i], nil
}

func (n *Neuron) MustGetConnection(i int) *Connection {
	c, err := n.GetConnection(i)
	if err != nil {
		panic(err)
	}
	return c
}

func (n *Neuron) GetConnections(i, j int) ([]*Connection, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return nil, fmt.Errorf("cannot get subset ending at > size of connections (requested j: %v)", j)
	}
	return n.Connections[i:j], nil
}

func (n *Neuron) MustGetConnections(i, j int) []*Connection {
	ons, err := n.GetConnections(i, j)
	if err != nil {
		panic(err)
	}
	return ons
}

func (n *Neuron) SetConnection(i int, c *Connection) error {
	if i < 0 {
		return fmt.Errorf("cannot set Connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return fmt.Errorf("cannot set Connection at index > size of connections, %v (requested %v)", len(n.Connections), i)
	}
	n.Connections[i] = c
	return nil
}

func (n *Neuron) MustSetConnection(i int, c *Connection) {
	err := n.SetConnection(i, c)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SetConnections(i, j int, cs []*Connection) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return fmt.Errorf("cannot set subset ending at > size of connections (requested j: %v)", j)
	}
	q := j - i
	if len(cs) != q {
		return fmt.Errorf("cannot set a subset of Connections to a set of Connections of different length. target subset length: %v, provided set length: %v", q, len(cs))
	}
	for k := 0; k < q; k++ {
		n.Connections[k+i] = cs[k]
	}
	return nil
}

func (n *Neuron) MustSetConnections(i, j int, cs []*Connection) {
	err := n.SetConnections(i, j, cs)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SwapOutConnection(i int, c *Connection) (*Connection, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return nil, fmt.Errorf("cannot swap connection at index > size of connections, %v (requested %v)", len(n.Connections), i)
	}
	on := n.MustGetConnection(i)
	n.MustSetConnection(i, c)
	return on, nil
}

func (n *Neuron) MustSwapOutConnection(i int, c *Connection) *Connection {
	oc, err := n.SwapOutConnection(i, c)
	if err != nil {
		panic(err)
	}
	return oc
}

func (n *Neuron) SwapOutConnections(i, j int, cs []*Connection) ([]*Connection, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of connections (requested j: %v)", j)
	}
	q := j - i
	if len(cs) != q {
		return nil, fmt.Errorf("cannot swap a subset of Connections with a set of Connections of different length. target subset length: %v, provided set length: %v", q, len(cs))
	}
	ols := n.MustGetConnections(i, j)
	n.MustSetConnections(i, j, cs)
	return ols, nil
}

func (n *Neuron) MustSwapOutConnections(i, j int, cs []*Connection) []*Connection {
	ons, err := n.SwapOutConnections(i, j, cs)
	if err != nil {
		panic(err)
	}
	return ons
}

func (n *Neuron) SetValue(value float64) {
	n.value = value
}

func (n *Neuron) SetLabel(label string) {
	n.label = label
}

func (n *Neuron) SetBias(bias float64) {
	n.bias = bias
}

func (n *Neuron) SetActivationFunction(activationFunctionName activationfunction.Name) error {
	af, err := activationfunction.GetFunction(activationFunctionName)
	if err != nil {
		return err
	}
	n.activationFunction = af
	n.ActivationFunctionName = activationFunctionName

	return nil
}

func (n *Neuron) MustSetActivationFunction(activationFunctionName activationfunction.Name) {
	err := n.SetActivationFunction(activationFunctionName)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SetConnectionWeights(weights []float64) error {
	if len(weights) != len(n.Connections) {
		return fmt.Errorf("invalid number of weights provided (%v), does not match number of connections in neuron (%v)", len(weights), len(n.Connections))
	}

	for ci := range n.Connections {
		n.Connections[ci].SetWeight(weights[ci])
	}
	return nil
}

func (n *Neuron) MustSetConnectionWeights(weights []float64) {
	err := n.SetConnectionWeights(weights)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SetConnectionWeightsTo(weight float64) {
	for ci := range n.Connections {
		n.Connections[ci].SetWeight(weight)
	}
}

//////////////////////////////////////// Connection ////////////////////////////////////////

func (c *Connection) Equals(c2 *Connection) error {
	// NOTE(justin): To avoid misleading behavior, nil values are not considered equal.
	if c == nil || c2 == nil {
		return fmt.Errorf("one or both of the connections are nil, %v || %v", c == nil, c2 == nil)
	}

	// NOTE(justin): This introduces a lot of calculations, exponentially so as the network grows in layers.
	// This line means that Neuron.Equals calls Connection.Equals which then calls Neuron.Equals again on its connecting
	// neuron. And while this will eventually terminate, it means there will be a lot of checks.
	if err := c.To.Equals(c2.To); err != nil {
		return err
	}

	if c.weight != c2.weight {
		return fmt.Errorf("connections' weights do not match, %v != %v", c.weight, c2.weight)
	}
	if c.dNetDWeight != c2.dNetDWeight {
		return fmt.Errorf("connections' dNetDWeights do not match, %v != %v", c.dNetDWeight, c2.dNetDWeight)
	}
	if c.dLossDWeight != c2.dLossDWeight {
		return fmt.Errorf("connections' dLossDWeights do not match, %v != %v", c.dLossDWeight, c2.dLossDWeight)
	}
	if c.dNetDPrevValue != c2.dNetDPrevValue {
		return fmt.Errorf("connections' dNetDPrevValues do not match, %v != %v", c.dNetDPrevValue, c2.dNetDPrevValue)
	}

	if len(c.weightNudges) != len(c2.weightNudges) {
		return fmt.Errorf("connections do not have same number of weight nudges, %v != %v", len(c.weightNudges), len(c.weightNudges))
	}
	for wi := range c.weightNudges {
		if c.weightNudges[wi] != c2.weightNudges[wi] {
			return fmt.Errorf("connections' biasNudges do not match, %v != %v", c.weightNudges[wi], c2.weightNudges[wi])
		}
	}

	return nil
}

func (c *Connection) SetWeight(weight float64) {
	c.weight = weight
}
