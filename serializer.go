package jnet

import (
	"fmt"
	"strconv"
	"strings"
)

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
		panic("Invalid version line!")
	}

	version := fl[0]

	lc++

	if version != 1.0 {
		panic("Invalid version number!")
	}

	// SECOND LINE
	sl := data[1]
	qnils := sl

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

	fmt.Printf("%#v\n", qnils)
	fmt.Printf("%#v\n", biases)
	fmt.Printf("%#v\n", weights)

	// CREATE NETWORK
	nw := &Network{}

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
