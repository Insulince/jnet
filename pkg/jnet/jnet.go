package jnet

import (
	"fmt"
	"math"
	"math/rand"
)

///////////////////////// TRAINING DATA /////////////////////////

type TrainingData struct {
	Data  []float64
	Truth []float64
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
	Value       float64 // Bounded to (0.0, 1.0) (This is sigmoid(WSum + Bias))
	WSum        float64 // Unbounded
	Bias        float64 // Unbounded
	Connections []*Connection
}

func NewNeuron(pl *Layer) (nn *Neuron) {
	nn = &Neuron{
		Bias: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}

	qpln := len(pl.Neurons)
	for ni := 0; ni < qpln; ni++ {
		pln := pl.Neurons[ni]

		nn.Connections = append(nn.Connections, NewConnection(pln, nn))
	}

	return nn
}

func (n *Neuron) Reset() (this *Neuron) {
	n.Value = 0.0
	n.WSum = 0.0

	return n
}

///////////////////////// CONNECTION /////////////////////////

type Connection struct {
	Left   *Neuron
	Weight float64 // Unbounded
	Right  *Neuron
}

func NewConnection(left *Neuron, right *Neuron) (c *Connection) {
	return &Connection{
		Left:   left,
		Right:  right,
		Weight: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
}

///////////////////////// LAYER /////////////////////////

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(qn int, pl *Layer) (nl *Layer) {
	for i := 0; i < qn; i++ {
		nl.Neurons = append(nl.Neurons, NewNeuron(pl))
	}
	return nl
}

func (l Layer) SetNeuronValues(values []float64) (this Layer) {
	qn := len(l.Neurons)

	if qn != len(values) {
		panic("Invalid number of values provided, does no match number of neurons in layer.")
	}

	for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
		l.Neurons[ni].Value = values[ni]
	}

	return l
}

///////////////////////// NETWORK /////////////////////////

const (
	LearningRate       = 0.1
	TrainingIterations = 50
	MiniBatchSize      = 3
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

// Predict will run an input vector, `input` against the neural network which sets its outputs for a prediction.
// Currently it just acts as an exported wrapper around forwardPass, because there is no need
// to do anything except a forward pass, however, to stay in line with BackwardPass, I made
// forwardPass unexported.
func (nw *Network) Predict(input []float64) (this *Network) {
	nw.forwardPass(input)

	return nw
}

// forwardPass runs the neural network against an input vector, `input`, and set its outputs for a prediction.
func (nw *Network) forwardPass(input []float64) (this *Network) {
	ql := len(nw.Layers)

	for li := 0; li < ql; li++ { // For every layer...
		l := nw.Layers[li]

		qn := len(l.Neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := l.Neurons[ni]

			n.Reset()
		}
	}

	nw.Layers[0].SetNeuronValues(input)

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

			n.Value = sigmoid(n.WSum + n.Bias)

			// TODO
			//// Local Gradients
			//for plni := 0; plni < qpln; plni++ { // For every neuron in the previous layer...
			//	pn := &pl[plni]
			//
			//	lg := n.AntiTransorm(n.Net) * pn.Output
			//	n.LocalGradients = append(n.LocalGradients, lg)
			//}
		}
	}

	return nw
}

// Train will exercise the network against a series of "truth" input vectors, `trainingData`, choosing one at random, `TrainingIterations` times.
// Each iteration, a forward pass is executed to get the current outputs, loss is calculated, a backward pass is executed to calculate the gradients,
// and each weight is adjusted to make the network perform better.
func (nw *Network) Train(trainingData []TrainingData) (this *Network) {
	fmt.Println("TRAINING START")

	for i := 0; i < TrainingIterations; i++ { // For every desired training iteration...
		// Shuffle current set of trainingData
		for i := range trainingData {
			j := rand.Intn(i + 1)
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		}

		// Slice out all trainingData elements we need.
		miniBatch := trainingData[0:MiniBatchSize]

		nw.Loss = 0.0

		for mbi := 0; mbi < MiniBatchSize; mbi++ {
			td := &miniBatch[mbi]

			nw.forwardPass(td.Data)
			nw.CalculateLoss(td.Truth)
		}

		nw.Loss /= MiniBatchSize // Get the average loss across the minibatch.

		nw.backwardPass()

		// TODO
		//nw.adjustWeights()

	}

	return nw
}

// BackwardPass runs the network backwards from the state its last forward pass left it in to determine the LossGradients of each neuron (by
// comparing to the provided loss value, `loss`), and adjusts each neuron's weight based on it to make the network perform better.
func (nw *Network) backwardPass() (this *Network) {
	ql := len(nw.Layers)
	fli, sli, lli, slli := 0, 1, ql-1, ql-2
	fl, sl, ll, sll := nw.Layers[fli], nw.Layers[sli], nw.Layers[lli], nw.Layers[slli]

	qlln := len(ll.Neurons)
	for llni := 0; llni < qlln; llni++ { // For every neuron in the last layer...
		//lln := &ll[llni]

		qslln := len(sll.Neurons)
		for sllni := 0; sllni < qslln; sllni++ { // For every neuron in the second to last layer...
			//slln := &sll[sllni]

			//lln.LossGradients = append(lln.LossGradients, nw.Loss*lln.AntiTransorm(slln.Output*slln.Connections[llni][connection.IndexWeight])*slln.Output)
		}

		//lln.LossGradientSum = lln.Loss * lln.AntiTransorm(lln.Net)
	}

	for li := slli; li > fli; li-- { // For every layer except the last and first, starting from the second to last...
		pl, l, nl := nw.Layers[li-1], nw.Layers[li], nw.Layers[li+1]

		qn := len(l.Neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			//n := &l[ni]

			qnln := len(nl.Neurons)
			for nlni := 0; nlni < qnln; nlni++ { // For every neuron in the next layer...
				//nln := &nl[nlni]

				qpln := len(pl.Neurons)
				for plni := 0; plni < qpln; plni++ { // For every neuron in the previous layer...
					//n.LossGradients = append(n.LossGradients, n.LocalGradients[plni]*nln.LossGradientSum)
				}

				//for _, lg := range n.LossGradients { // For every loss gradient in the current neuron...
				//n.LossGradientSum += lg
				//}
			}
		}
	}

	qfln := len(fl.Neurons)
	for flni := 0; flni < qfln; flni++ { // For every neuron in the first layer...
		//fln := &fl[flni]

		qsln := len(sl.Neurons)
		for slni := 0; slni < qsln; slni++ { // For every neuron in the second layer...
			//sln := &sl[slni]

			// We use fln.Value here instead of a local gradient because the first layer has only 1 local gradient, which is the input, thus the value.
			//fln.LossGradients = append(fln.LossGradients, sln.LossGradientSum*nw.Layers[sli+1][0].LossGradientSum)

			//for _, lg := range fln.LossGradients {
			//	fln.LossGradientSum += lg
			//}
		}
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
		nw.Loss += math.Pow(math.Abs(truth[ni]-n.Value), 2) // MSE
	}

	return nw
}
