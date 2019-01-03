package network

import (
	"fmt"
	"jnet/pkg/connection"
	"jnet/pkg/layer"
	"jnet/pkg/util"
)

type Network struct {
	*util.Logger
	Layers []layer.Layer
}

func New(ls []layer.Layer) (nnw *Network) {
	nw := &Network{
		Logger: util.NewLogger("Network", util.DefaultPadding),
		Layers: ls,
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
				cln.Connections = append(cln.Connections, *connection.New(clni, nlni))
			}
		}
	}

	return nw
}

func (nw *Network) RandomizeConnectionWeights() (this *Network) {
	for li := 0; li < len(nw.Layers)-1; li++ {
		cl := nw.Layers[li]
		for cni := 0; cni < len(cl); cni++ {
			cn := &cl[cni]
			for ci := 0; ci < len(cn.Connections); ci++ {
				cc := &cn.Connections[ci]

				cc[connection.IndexWeight] = connection.RandomWeight()
			}
		}
	}

	return nw
}

func (nw *Network) SetAllConnectionWeightsTo(v float64) (this *Network) {
	for li := 0; li < len(nw.Layers)-1; li++ {
		cl := nw.Layers[li]
		for cni := 0; cni < len(cl); cni++ {
			cn := &cl[cni]
			for ci := 0; ci < len(cn.Connections); ci++ {
				cc := &cn.Connections[ci]

				cc[connection.IndexWeight] = v
			}
		}
	}

	return nw
}

func (nw *Network) ApplyConnectionMaps(cms []connection.Map) (this *Network) {
	if len(nw.Layers) != len(cms)+1 {
		panic("Different number of layers than connection maps!")
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

func (nw *Network) Process() (this *Network) {
	// For every layer except the first...
	for li := 1; li < len(nw.Layers); li++ {
		l := nw.Layers[li]
		pli := li - 1
		pl := nw.Layers[pli]

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

	return nw
}

func (nw *Network) Results() (results string) {
	for _, v := range nw.Layers[len(nw.Layers)-1] {
		results = fmt.Sprintf("%v%10s %8f\n", results, v.Result, v.Value)
	}

	return results
}
