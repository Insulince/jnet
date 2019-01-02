package network

import (
	"jnet/pkg/connection"
	"jnet/pkg/layer"
	"jnet/pkg/util"
)

type Network struct {
	*util.Logger
	Layers []layer.Layer
}

func New(ls []layer.Layer) (nnw *Network) {
	return &Network{
		Logger: util.NewLogger("Network", util.DefaultPadding),
		Layers: ls,
	}
}

func (nw *Network) Connect(cms []connection.Map) (this *Network) {
	if len(nw.Layers) != len(cms)+1 {
		panic("Different number of layers than connection maps!")
	}

	for li := 0; li < len(nw.Layers)-1; li++ {
		cl := nw.Layers[li]
		nl := nw.Layers[li+1]

		// For every neuron in the current layer...
		for clni := 0; clni < len(cl); clni++ {
			cln := &cl[clni]

			// For every neuron in the next layer...
			for nlni := 0; nlni < len(nl); nlni++ {
				// Create a connection from the current current layer neuron to the current next layer neuron with weight 0.
				cln.Connections = append(cln.Connections, connection.Connection{float64(clni), float64(nlni), 0})
			}
		}
	}

	for cmsi := 0; cmsi < len(cms); cmsi++ {
		cm := cms[cmsi]

		for ci := 0; ci < len(cm); ci++ {
			c := cm[ci]

			ni := c[connection.IndexFrom]
			ci := c[connection.IndexTo]
			w := c[connection.IndexWeight]

			nw.Layers[cmsi][(int)(ni)].Connections[(int)(ci)][connection.IndexWeight] = w
		}
	}

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

				v := pn.Value * (&nw.Layers[pli][plni].Connections[lni])[connection.IndexWeight]
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

			v := slln.Value * (&nw.Layers[slli][sllni].Connections[llni])[connection.IndexWeight]

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
