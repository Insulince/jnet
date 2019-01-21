package jnet

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

func Deserialize(networkString string) (nw *Network, err error) {
	var data [][]float64

	lines := strings.Split(networkString, "\n")
	ql := len(lines)
	for li := 0; li < ql-1; li++ {
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

	ll := lines[ql-1]

	lc := 0

	// FIRST LINE
	fl := data[0]

	if len(fl) != 1 {
		return nil, errors.New("Invalid version line!")
	}

	version := fl[0]

	lc++

	if version != 1.0 {
		return nil, errors.New("Invalid version number!")
	}

	// SECOND LINE
	sl := data[1]
	nm := sl

	lc++

	// BIASES LINES
	var biases [][]float64
	qnm := len(nm)
	for i := 1; i < qnm; i++ {
		biases = append(biases, []float64{})

		line := data[lc]

		qn := int(nm[i])
		if len(line) != qn {
			return nil, errors.New("Different number of biases than expected!")
		}

		bi := i - 1
		for j := 0; j < qn; j++ {
			biases[bi] = append(biases[bi], line[j])
		}

		lc++
	}

	// WEIGHTS LINES
	var weights [][][]float64
	for i := 1; i < qnm; i++ {
		weights = append(weights, [][]float64{})

		qn := int(nm[i])
		for j := 0; j < qn; j++ {
			weights[i-1] = append(weights[i-1], []float64{})

			line := data[lc]

			qpn := int(nm[i-1])
			for k := 0; k < qpn; k++ {
				weights[i-1][j] = append(weights[i-1][j], line[k])
			}

			lc++
		}
	}

	// LAST LINE
	if lc != ql-1 {
		panic("?")
	}
	labels := strings.Split(ll, " ")

	return DeserializeF(nm, biases, weights, labels)
}

func DeserializeF(nm []float64, biases [][]float64, weights [][][]float64, labels []string) (nw *Network, err error) {
	// CREATE NETWORK
	nw = &Network{}

	// BUILD LAYERS & INITIAL CONNECTIONS
	qnm := len(nm)
	for i := 0; i < qnm; i++ {
		if i == 0 {
			nw.layers = append(nw.layers, newLayer(int(nm[i]), nil))
		} else {
			pl := nw.layers[i-1]
			nw.layers = append(nw.layers, newLayer(int(nm[i]), pl))
		}
	}

	// SETUP WEIGHTS AND BIASES
	ql := len(nw.layers)
	for li := 1; li < ql; li++ {
		l := nw.layers[li]

		// BIASES
		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ {
			n := l.neurons[ni]

			n.bias = biases[li-1][ni]
		}

		// WEIGHTS
		for ni := 0; ni < qn; ni++ {
			n := l.neurons[ni]

			qc := len(n.connections)
			for ci := 0; ci < qc; ci++ {
				c := n.connections[ci]

				c.weight = weights[li-1][ni][ci]
			}
		}
	}

	// SETUP OUTPUT LABELS
	qnl := len(labels)
	if qnl != int(nm[qnm-1]) {
		panic("Cannot deserialize, number of labels and output neurons do not match!")
	}
	ll := nw.layers[ql-1]
	for nli := 0; nli < qnl; nli++ {
		label := labels[nli]

		ll.neurons[nli].label = label
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
		    0 1 2 3
	*/

	// FIRST LINE
	networkString = fmt.Sprintf("%v\n", SerializeVerion)

	// SECOND LINE
	ql := len(nw.layers)
	for li := 0; li < ql; li++ {
		l := nw.layers[li]

		networkString += fmt.Sprintf("%v", len(l.neurons))
		if li < ql-1 {
			networkString += " "
		} else {
			networkString += "\n"
		}
	}

	// BIASES LINES
	for li := 1; li < ql; li++ {
		l := nw.layers[li]

		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ {
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
	for li := 0; li < ql; li++ {
		l := nw.layers[li]

		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ {
			n := l.neurons[ni]

			qc := len(n.connections)
			for ci := 0; ci < qc; ci++ {
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
	for llni := 0; llni < qlln; llni++ {
		lln := ll.neurons[llni]

		networkString += lln.label
		if llni < qlln-1 {
			networkString += " "
		}
	}

	return networkString
}
