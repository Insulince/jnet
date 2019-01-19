package jnet

import (
	"fmt"
	"github.com/TheDemx27/calculus"
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

	DcDa float64 // The effect this neuron's value has on the loss. <-
	DcDb float64 // The effect this neuron's bias has on the loss. = DcDa * DaDz * DzDb = DcDa * DaDz <-
	DaDz float64 // The effect this neuron's weighted sum + bias has on the neuron's value. ->
	DzDb float64 // The effect this neuron's bias has on the weighted sum + bias. (Always = 1.0) ->

	NudgeDcDb   []float64
	AverageDcDb float64
}

func NewNeuron(pl *Layer) (nn *Neuron) {
	nn = &Neuron{
		Bias: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}

	qpln := len(pl.Neurons)
	for ni := 0; ni < qpln; ni++ { // For every neuron in the previous layer...
		pln := pl.Neurons[ni]

		nn.Connections = append(nn.Connections, NewConnection(pln, nn))
	}

	return nn
}

func (n *Neuron) ResetForPass() (this *Neuron) {
	n.Value = 0.0
	n.WSum = 0.0

	n.DcDa = 0.0
	n.DcDb = 0.0
	n.DaDz = 0.0
	n.DzDb = 0.0

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.ResetForPass()
	}

	return n
}

func (n *Neuron) ResetForMiniBatch() (this *Neuron) {
	n.ResetForPass()

	n.NudgeDcDb = []float64{}
	n.AverageDcDb = 0.0

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.resetForMiniBatch()
	}

	return n
}

func (n *Neuron) recordNudges() (this *Neuron) {
	n.NudgeDcDb = append(n.NudgeDcDb, n.DcDb)

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.recordNudges()
	}

	return n
}

func (n *Neuron) averageNudges() (this *Neuron) {
	dcdbSum := 0.0
	for _, dcdb := range n.NudgeDcDb {
		dcdbSum += dcdb
	}

	n.AverageDcDb = dcdbSum / float64(len(n.NudgeDcDb))

	qc := len(n.Connections)
	for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer's neurons...
		c := n.Connections[ci]

		c.averageNudges()
	}

	return n
}

func (n *Neuron) adjustWeights(learningRate float64) (this *Neuron) {
	n.Bias -= n.AverageDcDb * learningRate

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
	Weight float64 // Unbounded
	Right  *Neuron

	DzDw  float64 // The effect this connection's weight has on the weighted sum + bias. ->
	DcDw  float64 // The effect this connection's weight has on the loss. = DcDa * DaDz * DzDw <-
	DzDa_ float64 // The effect this connection's left-neuron's activation has on the weighted sum + bias. ->

	NudgeDcDw   []float64
	AverageDcDw float64
}

func NewConnection(left *Neuron, right *Neuron) (c *Connection) {
	return &Connection{
		Left:   left,
		Right:  right,
		Weight: rand.Float64()*2 - 1, // Initialize randomly to [-1.0, 1.0)
	}
}

func (c *Connection) ResetForPass() (this *Connection) {
	c.DzDw = 0.0
	c.DcDw = 0.0
	c.DzDa_ = 0.0

	return c
}

func (c *Connection) resetForMiniBatch() (this *Connection) {
	c.ResetForPass()

	c.NudgeDcDw = []float64{}
	c.AverageDcDw = 0.0

	return c
}

func (c *Connection) recordNudges() (this *Connection) {
	c.NudgeDcDw = append(c.NudgeDcDw, c.DcDw)

	return c
}

func (c *Connection) averageNudges() (this *Connection) {
	dcdwSum := 0.0
	for _, dcdw := range c.NudgeDcDw {
		dcdwSum += dcdw
	}

	c.AverageDcDw = dcdwSum / float64(len(c.NudgeDcDw))

	return c
}

func (c *Connection) adjustWeights(learningRate float64) (this *Connection) {
	c.Weight -= c.AverageDcDw * learningRate

	return c
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
			n.DaDz = calculus.Diff(sigmoid, z)
			n.DzDb = 1.0

			for ci := 0; ci < qc; ci++ { // For every connection this neuron has to the the previous layer...
				c := n.Connections[ci]

				c.DzDw = c.Left.Value
				c.DzDa_ = c.Weight

				//         vvvvvv - loss gradient
				//c.DcDw = n.DcDa * n.DaDz * c.DzDw
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

		//averageLoss := totalLoss / float64(nw.MiniBatchSize) // Get the average loss across the whole minibatch.
		nw.averageNudges()

		nw.adjustWeights()
	}

	return nw
}

// BackwardPass runs the network backwards from the state its last forward pass left it in to determine the LossGradients of each neuron (by
// comparing to the provided loss value, `loss`), and adjusts each neuron's weight based on it to make the network perform better.
func (nw *Network) backwardPass(truth []float64) (this *Network) {
	//dcda := 2 * (ln.Value - truth)
	//dadz := calculus.Diff(sigmoid, ln.WSum+ln.Bias)
	//dzdw := nw.Layers[len(nw.Layers)-1].Neurons[0].Connections[0].Left.Value
	//dcdw := dcda * dadz * dzdw
	//
	//dzdb := 1.0
	//dcdb := dcda * dadz * dzdb
	//dcdb = dcda * dadz

	ll := nw.Layers[len(nw.Layers)-1]
	qlln := len(ll.Neurons)
	for llni := 0; llni < qlln; llni++ { // For every neuron in the last layer...
		lln := ll.Neurons[llni]

		lln.DcDa = 2 * (lln.Value - truth[llni])  // d(MSE)
		lln.DcDb = lln.DcDa * lln.DaDz * lln.DzDb // DzDb is always 1, so not really needed, but included for calculus reasons.

		qc := len(lln.Connections)
		for ci := 0; ci < qc; ci++ { // For every connection from this layer to its previous layer's neurons...
			c := lln.Connections[ci]
			c.DcDw = lln.DcDa * lln.DaDz * c.DzDw
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

				n.DcDa += nln.DcDa * nln.DaDz * nln.Connections[ni].DzDa_
			}

			n.DcDb = n.DcDa * n.DaDz * n.DzDb // DzDb is always 1, so not really needed, but included for calculus reasons.

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ { // For every connection from this neuron to its previous layer's neurons...
				c := n.Connections[ci]
				c.DcDw = n.DcDa * n.DaDz * c.DzDw
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
