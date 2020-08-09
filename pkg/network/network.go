package network

import (
	"errors"
	"fmt"
	"github.com/Insulince/jnet/pkg/train"
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
			nw = append(nw, newLayer(qn, nil))
			continue
		}
		pl := nw[li-1]
		nw = append(nw, newLayer(qn, pl))
	}

	if spec.InputLabels == nil {
		spec.InputLabels = make([]string, spec.NeuronMap[0])
	}
	err := nw.FirstLayer().setNeuronLabels(spec.InputLabels)
	if err != nil {
		return nil, err
	}

	if spec.OutputLabels == nil {
		return nil, errors.New("must provide output labels") // TODO(justin): Make optional?
	}
	err = nw.LastLayer().setNeuronLabels(spec.OutputLabels)
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
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j")
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

func (nw Network) SwapLayer(i int, l Layer) (Layer, error) {
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

func (nw Network) MustSwapLayer(i int, l Layer) Layer {
	ol, err := nw.SwapLayer(i, l)
	if err != nil {
		panic(err)
	}
	return ol
}

func (nw Network) SwapLayers(i, j int, ls []Layer) ([]Layer, error) {
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

func (nw Network) MustSwapLayers(i, j int, ls []Layer) []Layer {
	ols, err := nw.SwapLayers(i, j, ls)
	if err != nil {
		panic(err)
	}
	return ols
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

func (nw Network) Predict(input []float64) (string, error) {
	if len(input) != len(nw.FirstLayer()) {
		return "", fmt.Errorf("invalid number of values provided (%v), does no match number of neurons in Layer (%v)", len(input), len(nw.FirstLayer()))
	}

	nw.resetForPass()

	err := nw.forwardPass(input)
	if err != nil {
		return "", err
	}

	return nw.getHighestConfidenceNeuron().label, nil
}

func (nw Network) forwardPass(input []float64) (err error) {
	fl := nw[0]
	err = fl.setNeuronValues(input)
	if err != nil {
		return err
	}

	ql := len(nw)
	for li := 1; li < ql; li++ { // For every layer except the first, starting from the second...
		l := nw[li]

		qn := len(l)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := l[ni]

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.Connections[ci]

				n.wSum += c.left.value * c.weight
			}

			net := n.wSum + n.bias
			n.value = sigmoid(net)
			n.dValueDNet = calculus.Diff(sigmoid, net)
			n.dNetDBias = 1.0

			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.Connections[ci]

				c.dNetDWeight = c.left.value
				c.dNetDPrevValue = c.weight
			}
		}
	}

	return nil
}

func (nw Network) Train(td train.Data, tc train.Configuration) error {
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

			err := nw.forwardPass(td.Data)
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

func (nw Network) recordNudges() {
	for li := range nw {
		nw[li].recordNudges()
	}
}

func (nw Network) adjustWeights(learningRate float64) {
	ql := len(nw)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw[li]

		l.adjustWeights(learningRate)
	}
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

func (nw Network) getHighestConfidenceNeuron() *Neuron {
	ll := nw.LastLayer()
	var hcn = ll[0]
	for ni := range nw.LastLayer() {
		if ll[ni].value > hcn.value {
			hcn = ll[ni]
		}
	}
	return hcn
}
