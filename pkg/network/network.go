package network

import (
	"fmt"
	"jnet/pkg/connection"
	"jnet/pkg/data"
	"jnet/pkg/layer"
	"jnet/pkg/util"
	"math"
	"math/rand"
)

const (
	LearningRate         data.V        = 0.01
	StartingLossGradient data.Gradient = 1.0
	TrainingIterations                 = 20000
)

type Network struct {
	*util.Logger
	Layers []layer.Layer
	Loss   data.Loss
}

// New creates a new network made of the provided Layers, `layers`.
// It then calls createConnections to initialize the network connections.
func New(layers []layer.Layer) (this *Network) {
	nw := &Network{
		Logger: util.NewLogger("Network", util.DefaultPadding),
		Layers: layers,
	}
	nw.createConnections()

	return nw
}

// createConnections creates and initializes all connections between each consecutive layer to the default weight of 0.0.
func (nw *Network) createConnections() (this *Network) {
	ql := len(nw.Layers)
	for li := 0; li < ql-1; li++ { // For every layer except the last...
		l, nl := nw.Layers[li], nw.Layers[li+1]

		qn := len(l)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := &l[ni]

			qnln := len(nl)
			for nlni := 0; nlni < qnln; nlni++ { // For every neuron in the next layer...
				// Create a connection from the current neuron to the current next layer neuron (default weight 0.0).
				n.Connections = append(n.Connections, *connection.New(ni, nlni))
			}
		}
	}

	return nw
}

// RandomizeConnectionWeights goes through all connections in the network and sets all weights to random values.
func (nw *Network) RandomizeConnectionWeights() (this *Network) {
	ql := len(nw.Layers)
	for li := 0; li < ql-1; li++ { // For every layer except the last...
		l := nw.Layers[li]

		qn := len(l)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := &l[ni]

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ { // For every connection originating from this neuron...
				c := &n.Connections[ci]

				c[connection.IndexWeight] = connection.RandomWeight()
			}
		}
	}

	return nw
}

// SetAllConnectionWeightsTo goes through all connections in the network and sets all weights to `value`.
func (nw *Network) SetAllConnectionWeightsTo(value data.V) (this *Network) {
	ql := len(nw.Layers)
	for li := 0; li < ql-1; li++ { // For every layer except the last...
		l := nw.Layers[li]

		qn := len(l)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := &l[ni]

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ { // For every connection originating from this neuron...
				c := &n.Connections[ci]

				c[connection.IndexWeight] = value
			}
		}
	}

	return nw
}

// ApplyConnectionMaps takes in a slice of connection.Map, `connectionMaps`, and
// updates all connections in the network to match the details in `connectionMaps`.
// This is useful for pre-configuring a network to your liking, instead of using
// `SetAllConnectionWeightsTo` or `RandomizeConnectionWeights`.
// Panics if you did not provide the same number of connection maps as there are layers minus 1.
// This is because you can't half configure a network, you need to provide all details.
func (nw *Network) ApplyConnectionMaps(connectionMaps []connection.Map) (this *Network) {
	ql, qcm := len(nw.Layers), len(connectionMaps)

	if ql != qcm+1 {
		panic("Different number of layers than connection maps!")
	}

	for cmi := 0; cmi < qcm; cmi++ { // For every connection map in `connectionMaps`...
		cm, l := connectionMaps[cmi], nw.Layers[cmi]

		qc := len(cm)
		for ci := 0; ci < qc; ci++ { // For every connection in this connection map...
			c := &cm[ci]

			ni, ci, w := int(c[connection.IndexFrom]), int(c[connection.IndexTo]), c[connection.IndexWeight]

			// This line looks complex, but can be read as follows:
			// Get the layer corresponding to the current connectionMap, `l` (there is 1 connection map for each layer except the last. They correspond likewise).
			// Get the neuron in that layer that is being connected to something, `ni`, which comes from `c[connection.IndexFrom]`.
			// Get that neuron's connections.
			// Get the index of the connection that corresponds to the neuron we are going to connect to, `ci`, which comes from `c[connection.IndexTo]` (there is 1 connection for each neuron in the next layer. They correspond likewise).
			// Get the current weight of that connection, the value in the connection at `connection.IndexWeight`.
			// Assign it to the desired weight, `w`, which comes from `c[connection.IndexWeight]`.
			l[ni].Connections[ci][connection.IndexWeight] = w
		}
	}

	return nw
}

// CalculateLoss determines how far off the network's current output state is from the input vector, `truth`.
// Loss is stored on the network itself.
func (nw *Network) CalculateLoss(truth data.T) (this *Network) {
	ll := nw.Layers[len(nw.Layers)-1]
	qt, qn := len(truth), len(ll)

	if qt != qn {
		panic("Can't calculate loss, truth and output layer are of different lengths!")
	}

	nw.Loss = 0.0

	for ni := 0; ni < qn; ni++ { // For every neuron in the last layer...
		n := &ll[ni]
		nw.Loss += data.Loss(math.Pow(float64(truth[ni]-n.Value), 2)) // MSE
	}

	return nw
}

// Predict will run an input vector, `input` against the neural network which sets its outputs for a prediction.
// Currently it just acts as an exported wrapper around forwardPass, because there is no need
// to do anything except a forward pass, however, to stay in line with backwardPass, I made
// forwardPass unexported.
func (nw *Network) Predict(input data.D) (this *Network) {
	nw.forwardPass(input)

	return nw
}

// forwardPass runs the neural network against an input vector, `input`, and set its outputs for a prediction.
// During this process each neuron's `Value`, `Sum`, and `LocalGradients` are evaluated.
func (nw *Network) forwardPass(input data.D) (this *Network) {
	nw.Layers[0].SetInputNeuronValues(input)

	qn := len(nw.Layers)
	for li := 1; li < qn; li++ { // For every layer except the first, starting from the second...
		l, pli := nw.Layers[li], li-1
		pl := nw.Layers[pli]

		qn := len(l)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := &l[ni]

			n.StageForPass()

			qpln := len(pl)
			for plni := 0; plni < qpln; plni++ { // For every neuron in the previous layer...
				pn := &pl[plni]

				n.Sum += pn.Value * data.V((&nw.Layers[pli][plni].Connections[ni])[connection.IndexWeight])
			}

			n.Value = n.Transform(n.Sum)

			/*
				y = sin(a * b + c * d)
				dy/da = cos(b * a + c * d) * b
				which is because.....
				d(sin(2 * x)) = cos(2 * x) * d(2 * x) = cos(2 * x) * 2

				n_1_0's value is the sigmoid of the weighted sum of all n_0_x's values.

				n_1_0.sum = n_0_0.v * w_0_0
						  + n_0_1.v * w_0_1
						  + n_0_2.v * w_0_2
						  + n_0_3.v * w_0_3

				n_1_0.v = sig(n_1_0.sum)

				Inverse:
				"The change in n_1_0.v when w_0_x changes"
				d(w_0_x)/d(n_1_0.v) = d(sig(n_1_0.sum)) = dSig(n_1_0.sum) * d(n_1_0.sum) = dSig(n_1_0.sum) * n_0_0.v

				d(w_0_0)/d(n_1_0.sum) = n_0_0.v

				d(sig(...))/d(n_1_0.v) = dSig(...)

				dSig(n_1_0.v) =
					  n_0_0.v * w_0_0
					+ n_0_1.v * w_0_1
					+ n_0_2.v * w_0_2
					+ n_0_3.v * w_0_3

				d(w_0_0)/d(dSig(n_1_0.v)) = n_0_0.v

				d(w_0_0)/d(n_1_0.v) = d(w_0_0)/d() * d()/d(n_1_0.v)

				CHAIN:
				d(sig(...))/d() =

				y = x
				dx/dy = 1

				y = x + z
				dx/dy = 1
				dz/dy = 1

				y = x * z
				dx/dy = z
				dz/dy = x

				y = x * z + w
				dx/dy = z
				dz/dy = x
				dw/dy = 1

				y =   a * b
					+ c * d
					+ e * f
					+ g * h
				da/dy = b
				db/dy = a
				dc/dy = d
				dd/dy = c
				de/dy = f
				df/dy = e
				dg/dy = h
				dh/dy = g
			*/

			// Local Gradients
			for plni := 0; plni < qpln; plni++ { // For every neuron in the previous layer...
				pn := &pl[plni]

				lg := data.Gradient(n.AntiTransorm(n.Sum) * pn.Value)
				n.LocalGradients = append(n.LocalGradients, lg)
			}
		}
	}

	return nw
}

// Train will exercise the network against a series of "truth" input vectors, `trainingData`, choosing one at random, `TrainingIterations` times.
// Each iteration, a forward pass is executed to get the current outputs, loss is calculated, a backward pass is executed to calculate the gradients,
// and each weight is adjusted to make the network perform better.
func (nw *Network) Train(trainingData []data.TrainingData) (this *Network) {
	for i := 0; i < TrainingIterations; i++ { // For every desired training iteration...
		td := trainingData[rand.Intn(len(trainingData))]
		nw.forwardPass(td.Data)
		nw.CalculateLoss(td.Truth)
		fmt.Printf("%2v %v\n", td, nw.Loss)
		nw.backwardPass()
	}

	return nw
}

// TODO: This used to accept `loss`, but never used it????
// backwardPass runs the network backwards from the state its last forward pass left it in to determine the LossGradient of each neuron (by
// comparing to the provided loss value, `loss`), and adjusts each neuron's weight based on it to make the network perform better.
func (nw *Network) backwardPass() (this *Network) {
	lli := len(nw.Layers) - 1
	ll := nw.Layers[lli]

	qlln := len(ll)
	for llni := 0; llni < qlln; llni++ { // For every neuron in the last layer...
		lln := &ll[llni]

		// d(L)/d(n_i_j.v) = d(1)/(n_i_j.v) * d(L)/(1)
		lln.LossGradient += lln.LocalGradients[llni] * StartingLossGradient
	}

	for li := lli - 1; li > -1; li-- { // For every layer except the last, starting from the second to last...
		l, nl := nw.Layers[li], nw.Layers[li+1]

		qn := len(l)
		for ni := 0; ni < qn; ni++ { // For every neuron in the current layer...
			n := &l[ni]

			qnln := len(nl)
			for nlni := 0; nlni < qnln; nlni++ { // For every neuron in the next layer...
				nln := &nl[nlni]

				// d(L)/d(n_i_j.v) = d(n_i+1_j.v)/(n_i_j.v) * d(L)/(n_i+1_j.v)
				n.LossGradient += nln.LossGradient * nln.LocalGradients[ni]
			}

			qc := len(n.Connections)
			for ci := 0; ci < qc; ci++ { // For every connection from this neuron...
				c := &n.Connections[ci]

				c[connection.IndexWeight] -= data.V(n.LossGradient) * LearningRate
			}
		}
	}

	return nw
}

// GetResults takes the current state of the network, looks through the last layer, and outputs
// each neuron's value and corresponding result. Following that it outputs the result that had
// the highest value, which can be interpreted as the network's prediction.
func (nw *Network) GetResults() (results string) {
	maxValue, maxResult, ql := data.V(-1*math.MaxFloat64), "", len(nw.Layers)
	ll := nw.Layers[ql-1]

	qn := len(ll)
	for ni := 0; ni < qn; ni++ { // For every neuron in the last layer...
		n := ll[ni]

		results = fmt.Sprintf("%v%10s %8f\n", results, n.Result, n.Value)

		if n.Value > maxValue {
			maxValue, maxResult = n.Value, n.Result
		}
	}

	results = fmt.Sprintf("%vResult: %v\n", results, maxResult)

	return results
}
