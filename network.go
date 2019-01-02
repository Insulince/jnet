package jnet

type Network struct {
	*Logger
	Layers []Layer
}

func NewNetwork() (nnw *Network) {
	return &Network{
		Logger: NewLogger("Network", DefaultPadding),
		Layers: []Layer{},
	}
}

func (nw *Network) CreateConnections(l Layer) (this *Network) {
	pli := len(nw.Layers) - 1

	if pli > -1 {
		pl := &nw.Layers[pli]

		// For every neuron in the previous layer...
		for pni := 0; pni < len(*pl); pni++ {
			pn := &(*pl)[pni]

			// For every neuron in the current layer...
			for ni := 0; ni < len(l); ni++ {
				//n := l[ni]

				// Append the current neuron as a connection to the previous neuron with weight 0.
				pn.Connections = append(pn.Connections, *NewConnection())
			}
		}
	}

	nw.Layers = append(nw.Layers, l)

	return nw
}

func (nw *Network) SetConnectionWeight(pni int, pnci int, w float64) (this *Network) {
	pli := len(nw.Layers) - 2
	if pli < 0 {
		panic("Trying to set a connection weight on a layer which contains no connections!")
	}
	c := nw.getConnection(pli, pni, pnci)
	c.Weight = w

	return nw
}

func (nw *Network) SetOutputNeuronResult(ni int, r string) (this *Network) {
	cli := len(nw.Layers) - 1
	if cli < 0 {
		panic("Trying to set the output result of a non-output layer!")
	}
	nw.Layers[cli][ni].Result = r

	return nw
}

func (nw *Network) Process() (result string) {
	lli := len(nw.Layers) - 1
	ll := nw.Layers[lli]
	slli := len(nw.Layers) - 2
	sll := nw.Layers[slli]

	// For every layer except the first...
	for li := 1; li < lli; li++ {
		l := nw.Layers[li]
		pl := nw.Layers[li-1]
		pli := li - 1

		// For every neuron in the current layer...
		for lni := 0; lni < len(l); lni++ {
			n := &l[lni]

			// For every neuron in the previous layer...
			for plni := 0; plni < len(pl); plni++ {
				pn := &pl[plni]

				v := pn.Value * nw.getConnection(pli, plni, lni).Weight
				n.Value += v
			}

			n.Value = n.Transform(n.Value)
		}
	}

	// For every neuron in the last layer...
	for llni := 0; llni < len(ll); llni++ {
		lln := &ll[llni]

		// For every neuron in the second to last layer...
		for sllni := 0; sllni < len(sll); sllni++ {
			slln := &sll[sllni]

			v := slln.Value * nw.getConnection(slli, sllni, llni).Weight
			lln.Value += v
		}

		lln.Value = lln.Transform(lln.Value)
	}

	// For every neuron in the last layer...
	for llni := 0; llni < len(ll); llni++ {
		lln := &ll[llni]
		if lln.Value == 1 {
			if result != "" {
				panic("Result was set twice!")
			}
			result = nw.Layers[lli][llni].Result
		}
	}

	if result == "" {
		panic("Result was never set!")
	}

	return result
}

func (nw *Network) getConnection(li int, ni int, ci int) (c *Connection) {
	return &nw.Layers[li][ni].Connections[ci]
}
