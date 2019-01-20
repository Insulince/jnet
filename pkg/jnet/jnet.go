package jnet

import (
	"fmt"
	"github.com/TheDemx27/calculus"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

///////////////////////// TRAINING DATA /////////////////////////

type TrainingData struct {
	Data  []float64
	Truth []float64
}

///////////////////////// HUMAN DATA /////////////////////////

type HumanData struct {
	Data  [][]float64
	Truth []float64
}

func (hd *HumanData) ToTrainingData() (td *TrainingData) {
	td = &TrainingData{}

	qdr := len(hd.Data)
	for dri := 0; dri < qdr; dri++ {
		dr := hd.Data[dri]

		qd := len(dr)
		for di := 0; di < qd; di++ {
			datum := dr[di]

			td.Data = append(td.Data, datum)
		}
	}

	td.Truth = hd.Truth

	return td
}

///////////////////////// ACTIVATION FUNCTIONS /////////////////////////

func sigmoid(x float64) (y float64) {
	return 1 / (1 + math.Pow(math.E, -x))
}

func tanh(x float64) (y float64) {
	return math.Tanh(x)
}

func relu(x float64) (y float64) {
	if x > 0 {
		return x
	}
	return 0
}

func linear(x float64) (y float64) {
	return x
}

///////////////////////// NEURON /////////////////////////

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

///////////////////////// CONNECTION /////////////////////////

type Connection struct {
	Left   *Neuron
	Weight float64
	Right  *Neuron

	dNetDWeight    float64 // The effect this connection's weight has on the weighted sum + bias. ->
	dLossDWeight   float64 // The effect this connection's weight has on the loss. = dLossDValue * dValueDNet * dNetDWeight <-
	dNetDPrevValue float64 // The effect this connection's left-neuron's activation has on the weighted sum + bias. ->

	WeightNudges       []float64
	AverageWeightNudge float64
}

func NewConnection(left *Neuron, right *Neuron) (c *Connection) {
	return &Connection{
		Left:   left,
		Right:  right,
		Weight: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
}

func (c *Connection) ResetForPass() (this *Connection) {
	c.dNetDWeight = 0.0
	c.dLossDWeight = 0.0
	c.dNetDPrevValue = 0.0

	return c
}

func (c *Connection) resetForMiniBatch() (this *Connection) {
	c.ResetForPass()

	c.WeightNudges = []float64{}
	c.AverageWeightNudge = 0.0

	return c
}

func (c *Connection) recordNudges() (this *Connection) {
	c.WeightNudges = append(c.WeightNudges, c.dLossDWeight)

	return c
}

func (c *Connection) averageNudges() (this *Connection) {
	dcdwSum := 0.0
	for _, dcdw := range c.WeightNudges {
		dcdwSum += dcdw
	}

	c.AverageWeightNudge = dcdwSum / float64(len(c.WeightNudges))

	return c
}

func (c *Connection) adjustWeights(learningRate float64) (this *Connection) {
	c.Weight -= c.AverageWeightNudge * learningRate

	return c
}

///////////////////////// LAYER /////////////////////////

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(qn int, pl *Layer) (nl *Layer) {
	nl = &Layer{}

	for i := 0; i < qn; i++ {
		nl.Neurons = append(nl.Neurons, NewNeuron(pl))
	}
	return nl
}

func (l *Layer) SetNeuronValues(values []float64) (this *Layer) {
	qn := len(l.Neurons)

	if qn != len(values) {
		panic("Invalid number of values provided, does no match number of neurons in layer.")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.Neurons[ni].Value = values[ni]
	}

	return l
}

func (l *Layer) resetForPass() (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ {
		n := l.Neurons[ni]

		n.ResetForPass()
	}

	return l
}

func (l *Layer) resetForMiniBatch() (this *Layer) {
	l.resetForPass()

	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ {
		n := l.Neurons[ni]

		n.ResetForMiniBatch()
	}

	return l
}

func (l *Layer) recordNudges() (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.Neurons[ni]

		n.recordNudges()
	}

	return l
}

func (l *Layer) averageNudges() (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.Neurons[ni]

		n.averageNudges()
	}

	return l
}

func (l *Layer) adjustWeights(learningRate float64) (this *Layer) {
	qn := len(l.Neurons)
	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		n := l.Neurons[ni]

		n.adjustWeights(learningRate)
	}

	return l
}

///////////////////////// NETWORK /////////////////////////

type Network struct {
	Layers []*Layer
	Loss   float64

	LearningRate       float64
	TrainingIterations int
	MiniBatchSize      int
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

				//         vvvvvv - loss gradient
				//c.dLossDWeight = n.dLossDValue * n.dValueDNet * c.dNetDWeight
				//   local gradient ^^^^^^^^^^^^^^^
			}
		}
	}

	return nw
}

// Train will exercise the network against a series of "truth" input vectors, `trainingData`, choosing one at random, `TrainingIterations` times.
// Each iteration, a forward pass is executed to get the current outputs, loss is calculated, a backward pass is executed to calculate the gradients,
// and each weight is adjusted to make the network perform better.
func (nw *Network) Train(trainingData []TrainingData) (this *Network) {
	fmt.Println("TRAINING START")

	c := 0

	totalAverageLoss := 0.0
	averageAverageLoss := 0.0
	minAverageLoss := float64(math.MaxInt32)
	maxAverageLoss := float64(-math.MaxInt32)

	// TODO: Stop this when sufficiently minimized.
	for i := 0; i < nw.TrainingIterations; i++ { // For every desired training iteration...
		// Shuffle current set of trainingData
		for i := range trainingData {
			j := rand.Intn(i + 1)
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		}

		// Slice out all trainingData elements we need.
		miniBatch := trainingData[0:nw.MiniBatchSize]

		totalLoss := 0.0

		nw.resetForMiniBatch()
		for mbi := 0; mbi < nw.MiniBatchSize; mbi++ {
			td := &miniBatch[mbi]

			nw.resetForPass()
			nw.forwardPass(td.Data)

			nw.CalculateLoss(td.Truth)
			totalLoss += nw.Loss

			nw.backwardPass(td.Truth)

			nw.recordNudges()
		}

		c++
		averageLoss := totalLoss / float64(nw.MiniBatchSize) // Get the average loss across the whole minibatch.
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

		nw.adjustWeights()
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

func (nw *Network) adjustWeights() (this *Network) {
	ql := len(nw.Layers)

	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.Layers[li]

		l.adjustWeights(nw.LearningRate)
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

func Deserialize(networkString string) (this *Network, err error) {
	var data [][]float64

	lines := strings.Split(networkString, "\n")
	ql := len(lines)
	for li := 0; li < ql; li++ {
		line := lines[li]

		data = append(data, []float64{})

		items := strings.Split(line, " ")
		qi := len(items)
		for ii := 0; ii < qi; ii++ {
			item := items[ii]

			value, err := strconv.ParseFloat(item, 64)
			if err != nil {
				return nil, err
			}
			data[li] = append(data[li], value)
		}
	}

	lc := 0

	// FIRST LINE
	fl := data[0]

	if len(fl) != 1 {
		panic("Invalid version identifier!")
	}

	_ = fl[0]

	lc++

	// SECOND LINE
	sl := data[1]

	if len(sl) != 3 {
		panic("Different number of configuration values provided than expected!")
	}

	lr := sl[0]
	ti := sl[1]
	mbs := sl[2]

	lc++

	// THIRD LINE
	tl := data[2]
	qnils := tl

	lc++

	// BIASES LINES
	var biases [][]float64
	for i := 1; i < len(qnils); i++ {
		biases = append(biases, []float64{})

		line := data[lc]

		qn := int(qnils[i])
		if len(line) != qn {
			panic("Different number of biases than expected!")
		}

		bi := i - 1
		for j := 0; j < qn; j++ {
			biases[bi] = append(biases[bi], line[j])
		}

		lc++
	}

	defer func() {
		if err := recover(); err != nil {
			fmt.Println()
		}
	}()

	// WEIGHTS LINES
	var weights [][][]float64
	for i := 1; i < len(qnils); i++ {
		weights = append(weights, [][]float64{})

		qn := int(qnils[i])
		for j := 0; j < qn; j++ {
			weights[i-1] = append(weights[i-1], []float64{})

			line := data[lc]

			qpn := int(qnils[i-1])
			for k := 0; k < qpn; k++ {
				weights[i-1][j] = append(weights[i-1][j], line[k])
			}

			lc++
		}
	}

	fmt.Printf("%#v\n", lr)
	fmt.Printf("%#v\n", ti)
	fmt.Printf("%#v\n", mbs)
	fmt.Printf("%#v\n", qnils)
	fmt.Printf("%#v\n", biases)
	fmt.Printf("%#v\n", weights)

	// CREATE NETWORK
	nw := &Network{}

	// CONFIGURE
	nw.LearningRate = lr
	nw.TrainingIterations = int(ti)
	nw.MiniBatchSize = int(mbs)

	// BUILD LAYERS & INITIAL CONNECTIONS
	for i := 0; i < len(qnils); i++ {
		if i == 0 {
			nw.Layers = append(nw.Layers, NewLayer(int(qnils[i]), nil))
		} else {
			pl := nw.Layers[i-1]
			nw.Layers = append(nw.Layers, NewLayer(int(qnils[i]), pl))
		}
	}

	// SETUP WEIGHTS AND BIASES
	for li := 1; li < len(nw.Layers); li++ {
		l := nw.Layers[li]

		// BIASES
		for ni := 0; ni < len(l.Neurons); ni++ {
			n := l.Neurons[ni]

			n.Bias = biases[li-1][ni]
		}

		// WEIGHTS
		for ni := 0; ni < len(l.Neurons); ni++ {
			n := l.Neurons[ni]

			for ci := 0; ci < len(n.Connections); ci++ {
				c := n.Connections[ci]

				c.Weight = weights[li-1][ni][ci]
			}
		}
	}

	return nw, nil
}

const SerializeVerion = "1.0"

func (nw *Network) Serialize() (networkString string) {
	/*
		1.0
		0.1 2500000 32
		5 3 4
		1 2 3
		2 5 7 2
		1 1 1 1 1
		1 1 1 1 1
		1 1 1 1 1
		2 2 2
		2 2 2
		2 2 2
		2 2 2
	*/

	// FIRST LINE
	networkString = fmt.Sprintf("%v\n", SerializeVerion)

	// SECOND LINE
	networkString += fmt.Sprintf("%v %v %v\n", nw.LearningRate, nw.TrainingIterations, nw.MiniBatchSize)

	// THIRD LINE
	ql := len(nw.Layers)
	for li := 0; li < ql; li++ {
		l := nw.Layers[li]

		networkString += fmt.Sprintf("%v", len(l.Neurons))
		if li < ql-1 {
			networkString += " "
		} else {
			networkString += "\n"
		}
	}

	// BIASES LINES
	for li := 1; li < ql; li++ {
		l := nw.Layers[li]

		qn := len(l.Neurons)
		for ni := 0; ni < qn; ni++ {
			n := l.Neurons[ni]

			networkString += fmt.Sprintf("%v", n.Bias)

			if ni < qn-1 {
				networkString += " "
			} else {
				networkString += "\n"
			}
		}
	}

	// WEIGHTS LINES
	for li := 0; li < ql; li++ {
		l := nw.Layers[li]

		qn := len(l.Neurons)
		for ni := 0; ni < qn; ni++ {
			n := l.Neurons[ni]

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ {
				c := n.Connections[ci]

				networkString += fmt.Sprintf("%v", c.Weight)

				if ci < qc-1 {
					networkString += " "
				} else {
					if li < ql-1 || ni < qn-1 { // The string should not end with a newline, so if this is the last neuron, don't use a new line.
						networkString += "\n"
					}
				}
			}
		}
	}

	return networkString
}
