package jnet

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type deserializationData struct {
	neuronMap []float64
	biases    [][]float64
	weights   [][][]float64
	labels    []string
}

func Deserialize(networkString string) (nw *Network, err error) {
	dd, err := extractData(networkString)
	if err != nil {
		return nil, err
	}
	nw, err = dataToNetwork(dd)
	if err != nil {
		return nil, err
	}
	return nw, nil
}

func extractData(networkString string) (dd *deserializationData, err error) {
	dd = &deserializationData{}

	var data [][]float64

	lines := strings.Split(networkString, "\n")
	ql := len(lines)
	for li := 0; li < ql-1; li++ { // For every line in the network string...
		line := lines[li]

		data = append(data, []float64{})

		items := strings.Split(line, " ")
		qi := len(items)
		for ii := 0; ii < qi; ii++ { // For every item in this line...
			item := items[ii]

			value, err := strconv.ParseFloat(item, 64)
			if err != nil {
				return nil, err
			}
			data[li] = append(data[li], value)
		}
	}

	ll := lines[ql-1]

	lc := 0

	// FIRST LINE
	fl := data[0]

	if len(fl) != 1 {
		return nil, errors.New("invalid version line")
	}

	version := fl[0]

	lc++

	if version != 1.0 {
		return nil, errors.New("invalid version number")
	}

	// SECOND LINE
	sl := data[1]
	dd.neuronMap = sl

	lc++

	// BIASES LINES
	qnm := len(dd.neuronMap)
	for nmi := 1; nmi < qnm; nmi++ { // For every quantity of neurons in the neuron map...
		dd.biases = append(dd.biases, []float64{})

		line := data[lc]

		qn := int(dd.neuronMap[nmi])
		if len(line) != qn {
			return nil, errors.New("different number of biases than expected")
		}

		bi := nmi - 1
		for ni := 0; ni < qn; ni++ { // For every desired neuron in this layer...
			dd.biases[bi] = append(dd.biases[bi], line[ni])
		}

		lc++
	}

	// WEIGHTS LINES
	for nmi := 1; nmi < qnm; nmi++ { // For every quantity of neurons in the neuron map...
		dd.weights = append(dd.weights, [][]float64{})

		qn := int(dd.neuronMap[nmi])
		for ni := 0; ni < qn; ni++ { // For every desired neuron in this layer...
			dd.weights[nmi-1] = append(dd.weights[nmi-1], []float64{})

			line := data[lc]

			qpn := int(dd.neuronMap[nmi-1])
			for pni := 0; pni < qpn; pni++ { // For every desired neuron in the previous layer...
				dd.weights[nmi-1][ni] = append(dd.weights[nmi-1][ni], line[pni])
			}

			lc++
		}
	}

	// LAST LINE
	if lc != ql-1 {
		return nil, errors.New("invalid number of lines in network string")
	}
	dd.labels = strings.Split(ll, " ")

	return dd, nil
}

func dataToNetwork(dd *deserializationData) (nw *Network, err error) {
	// CREATE NETWORK
	nw = &Network{}

	// BUILD LAYERS & INITIAL CONNECTIONS
	qnm := len(dd.neuronMap)
	for nmi := 0; nmi < qnm; nmi++ { // For every quantity of neuron in the neuron map...
		qn := dd.neuronMap[nmi]

		if nmi == 0 {
			nw.layers = append(nw.layers, newLayer(int(qn), nil))
		} else {
			pl := nw.layers[nmi-1]
			nw.layers = append(nw.layers, newLayer(int(qn), pl))
		}
	}

	// SETUP WEIGHTS AND BIASES
	ql := len(nw.layers)
	for li := 1; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		// BIASES
		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
			n := l.neurons[ni]

			n.bias = dd.biases[li-1][ni]
		}

		// WEIGHTS
		for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
			n := l.neurons[ni]

			qc := len(n.connections)
			for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer...
				c := n.connections[ci]

				c.weight = dd.weights[li-1][ni][ci]
			}
		}
	}

	// SETUP OUTPUT LABELS
	qnl := len(dd.labels)
	if qnl != int(dd.neuronMap[qnm-1]) {
		return nil, errors.New("cannot deserialize, number of labels and output neurons do not match")
	}
	ll := nw.layers[ql-1]
	for nli := 0; nli < qnl; nli++ { // For every output label...
		label := dd.labels[nli]

		ll.neurons[nli].label = label
	}

	return nw, nil
}

const SerializeVerion = "1.0"

func (nw *Network) Serialize() (networkString string) {
	// FIRST LINE
	networkString = fmt.Sprintf("%v\n", SerializeVerion)

	// SECOND LINE
	ql := len(nw.layers)
	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		networkString += fmt.Sprintf("%v", len(l.neurons))
		if li < ql-1 {
			networkString += " "
		} else {
			networkString += "\n"
		}
	}

	// BIASES LINES
	for li := 1; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
			n := l.neurons[ni]

			networkString += fmt.Sprintf("%v", n.bias)

			if ni < qn-1 {
				networkString += " "
			} else {
				networkString += "\n"
			}
		}
	}

	// WEIGHTS LINES
	for li := 0; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
			n := l.neurons[ni]

			qc := len(n.connections)
			for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer...
				c := n.connections[ci]

				networkString += fmt.Sprintf("%v", c.weight)

				if ci < qc-1 {
					networkString += " "
				} else {
					networkString += "\n"
				}
			}
		}
	}

	// LAST LINE
	lli := ql - 1
	ll := nw.layers[lli]
	qlln := len(ll.neurons)
	for llni := 0; llni < qlln; llni++ { // For neuron in the last layer...
		lln := ll.neurons[llni]

		networkString += lln.label
		if llni < qlln-1 {
			networkString += " "
		}
	}

	return networkString
}
