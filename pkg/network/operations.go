package network

import "fmt"

//////////////////////////////////////// Network ////////////////////////////////////////

func (nw Network) Equals(nw2 Network) (bool, string) {
	if len(nw) != len(nw2) {
		return false, "networks are different lengths"
	}

	for li := range nw {
		if equal, reason := nw[li].Equals(nw2[li]); !equal {
			return false, reason
		}
	}

	return true, ""
}

func (nw Network) NumLayers() int {
	return len(nw)
}

func (nw Network) FirstLayer() Layer {
	return nw[0]
}

func (nw Network) LastLayer() Layer {
	return nw[len(nw)-1]
}

func (nw Network) GetLayer(i int) (Layer, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get layer at index < 0 (requested %v)", i)
	}
	if i >= len(nw) {
		return nil, fmt.Errorf("cannot get layer at index > size of network, %v (requested %v)", len(nw), i)
	}
	return nw[i], nil
}

func (nw Network) MustGetLayer(i int) Layer {
	l, err := nw.GetLayer(i)
	if err != nil {
		panic(err)
	}
	return l
}

// GetLayers returns the subset of layers from i to j, i inclusive and j exclusive.
func (nw Network) GetLayers(i, j int) ([]Layer, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(nw) {
		return nil, fmt.Errorf("cannot get subset ending at > size of network (requested j: %v)", j)
	}
	return nw[i:j], nil
}

func (nw Network) MustGetLayers(i, j int) []Layer {
	ls, err := nw.GetLayers(i, j)
	if err != nil {
		panic(err)
	}
	return ls
}

func (nw Network) SetLayer(i int, l Layer) error {
	if i < 0 {
		return fmt.Errorf("cannot set layer at index < 0 (requested %v)", i)
	}
	if i >= len(nw) {
		return fmt.Errorf("cannot set layer at index > size of network, %v (requested %v)", len(nw), i)
	}
	nw[i] = l
	if i > 0 {
		l.ConnectTo(nw[i-1])
	}
	return nil
}

func (nw Network) MustSetLayer(i int, l Layer) {
	err := nw.SetLayer(i, l)
	if err != nil {
		panic(err)
	}
}

// SetLayers sets the subset of layers from i to j, i inclusive and j exclusive, to ls.
func (nw Network) SetLayers(i, j int, ls []Layer) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(nw) {
		return fmt.Errorf("cannot set subset ending at > size of network (requested j: %v)", j)
	}
	q := j - i
	if len(ls) != q {
		return fmt.Errorf("cannot set a subset of layers to a set of layers of different length. target subset length: %v, provided set length: %v", q, len(ls))
	}
	for k := 0; k < q; k++ {
		nw[k+i] = ls[k]
		if k+i > 0 {
			ls[k].ConnectTo(nw[k+i-1])
		}
	}
	return nil
}

func (nw Network) MustSetLayers(i, j int, ls []Layer) {
	err := nw.SetLayers(i, j, ls)
	if err != nil {
		panic(err)
	}
}

func (nw Network) SwapOutLayer(i int, l Layer) (Layer, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap layer at index < 0 (requested %v)", i)
	}
	if i >= len(nw) {
		return nil, fmt.Errorf("cannot swap layer at index > size of network, %v (requested %v)", len(nw), i)
	}
	ol := nw.MustGetLayer(i)
	nw.MustSetLayer(i, l)
	return ol, nil
}

func (nw Network) MustSwapOutLayer(i int, l Layer) Layer {
	ol, err := nw.SwapOutLayer(i, l)
	if err != nil {
		panic(err)
	}
	return ol
}

func (nw Network) SwapOutLayers(i, j int, ls []Layer) ([]Layer, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(nw) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of network (requested j: %v)", j)
	}
	q := j - i
	if len(ls) != q {
		return nil, fmt.Errorf("cannot swap a subset of layers with a set of layers of different length. target subset length: %v, provided set length: %v", q, len(ls))
	}
	ols := nw.MustGetLayers(i, j)
	nw.MustSetLayers(i, j, ls)
	return ols, nil
}

func (nw Network) MustSwapOutLayers(i, j int, ls []Layer) []Layer {
	ols, err := nw.SwapOutLayers(i, j, ls)
	if err != nil {
		panic(err)
	}
	return ols
}

//////////////////////////////////////// Layer ////////////////////////////////////////

func (l Layer) Equals(l2 Layer) (bool, string) {
	if len(l) != len(l2) {
		return false, "layers are different lengths"
	}

	for ni := range l {
		if equal, reason := l[ni].Equals(l2[ni]); !equal {
			return false, reason
		}
	}

	return true, ""
}

func (l Layer) NumNeurons() int {
	return len(l)
}

func (l Layer) FirstNeuron() *Neuron {
	return l[0]
}

func (l Layer) LastNeuron() *Neuron {
	return l[len(l)-1]
}

func (l Layer) GetNeuron(i int) (*Neuron, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return nil, fmt.Errorf("cannot get neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	return l[i], nil
}

func (l Layer) MustGetNeuron(i int) *Neuron {
	n, err := l.GetNeuron(i)
	if err != nil {
		panic(err)
	}
	return n
}

func (l Layer) GetNeurons(i, j int) ([]*Neuron, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return nil, fmt.Errorf("cannot get subset ending at > size of layer (requested j: %v)", j)
	}
	return l[i:j], nil
}

func (l Layer) MustGetNeurons(i, j int) []*Neuron {
	ons, err := l.GetNeurons(i, j)
	if err != nil {
		panic(err)
	}
	return ons
}

func (l Layer) SetNeuron(i int, n *Neuron, pl Layer) error {
	if i < 0 {
		return fmt.Errorf("cannot set neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return fmt.Errorf("cannot set neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	l[i] = n
	if i > 0 {
		n.ConnectTo(pl)
	}
	return nil
}

func (l Layer) MustSetNeuron(i int, n *Neuron, pl Layer) {
	err := l.SetNeuron(i, n, pl)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SetNeurons(i, j int, ns []*Neuron, pl Layer) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return fmt.Errorf("cannot set subset ending at > size of layer (requested j: %v)", j)
	}
	q := j - i
	if len(ns) != q {
		return fmt.Errorf("cannot set a subset of neurons to a set of neurons of different length. target subset length: %v, provided set length: %v", q, len(ns))
	}
	for k := 0; k < q; k++ {
		l[k+i] = ns[k]
		if k+i > 0 {
			ns[k].ConnectTo(pl)
		}
	}
	return nil
}

func (l Layer) MustSetNeurons(i, j int, ns []*Neuron, pl Layer) {
	err := l.SetNeurons(i, j, ns, pl)
	if err != nil {
		panic(err)
	}
}

func (l Layer) SwapOutNeuron(i int, n *Neuron, pl Layer) (*Neuron, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap neuron at index < 0 (requested %v)", i)
	}
	if i >= len(l) {
		return nil, fmt.Errorf("cannot swap neuron at index > size of layer, %v (requested %v)", len(l), i)
	}
	on := l.MustGetNeuron(i)
	l.MustSetNeuron(i, n, pl)
	return on, nil
}

func (l Layer) MustSwapOutNeuron(i int, n *Neuron, pl Layer) *Neuron {
	on, err := l.SwapOutNeuron(i, n, pl)
	if err != nil {
		panic(err)
	}
	return on
}

func (l Layer) SwapOutNeurons(i, j int, ns []*Neuron, pl Layer) ([]*Neuron, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(l) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of layer (requested j: %v)", j)
	}
	q := j - i
	if len(ns) != q {
		return nil, fmt.Errorf("cannot swap a subset of neurons with a set of neurons of different length. target subset length: %v, provided set length: %v", q, len(ns))
	}
	ols := l.MustGetNeurons(i, j)
	l.MustSetNeurons(i, j, ns, pl)
	return ols, nil
}

func (l Layer) MustSwapOutNeurons(i, j int, ns []*Neuron, pl Layer) []*Neuron {
	ons, err := l.SwapOutNeurons(i, j, ns, pl)
	if err != nil {
		panic(err)
	}
	return ons
}

//////////////////////////////////////// Neuron ////////////////////////////////////////

func (n *Neuron) Equals(n2 *Neuron) (bool, string) {
	// NOTE(justin): To avoid misleading behavior, nil values are not considered equal.
	if n == nil || n2 == nil {
		return false, "one or both of the neurons are nil"
	}

	if len(n.Connections) != len(n2.Connections) {
		return false, "neurons do not have same number of connections"
	}
	for ci := range n.Connections {
		if equal, reason := n.Connections[ci].Equals(n2.Connections[ci]); !equal {
			return false, reason
		}
	}

	if n.label != n2.label {
		return false, "neurons' labels do not match"
	}
	if n.value != n2.value {
		return false, "neurons' values do not match"
	}
	if n.wSum != n2.wSum {
		return false, "neurons' wSums do not match"
	}
	if n.bias != n2.bias {
		return false, "neurons' biases do not match"
	}
	if n.dLossDValue != n2.dLossDValue {
		return false, "neurons' dLossDValues do not match"
	}
	if n.dLossDBias != n2.dLossDBias {
		return false, "neurons' dLossDBias do not match"
	}
	if n.dValueDNet != n2.dValueDNet {
		return false, "neurons' dValueDNet do not match"
	}
	if n.dNetDBias != n2.dNetDBias {
		return false, "neurons' dNetDBias do not match"
	}

	if len(n.biasNudges) != len(n2.biasNudges) {
		return false, "neurons do not have same number of bias nudges"
	}
	for bi := range n.biasNudges {
		if n.biasNudges[bi] != n2.biasNudges[bi] {
			return false, "neurons' biasNudges do not match"
		}
	}

	return true, ""
}

func (n *Neuron) NumConnections() int {
	return len(n.Connections)
}

func (n *Neuron) FirstConnection() *Connection {
	return n.Connections[0]
}

func (n *Neuron) LastConnection() *Connection {
	return n.Connections[len(n.Connections)-1]
}

func (n *Neuron) GetConnection(i int) (*Connection, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot get Connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return nil, fmt.Errorf("cannot get Connection at index > size of Neuron, %v (requested %v)", len(n.Connections), i)
	}
	return n.Connections[i], nil
}

func (n *Neuron) MustGetConnection(i int) *Connection {
	c, err := n.GetConnection(i)
	if err != nil {
		panic(err)
	}
	return c
}

func (n *Neuron) GetConnections(i, j int) ([]*Connection, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot get subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot get subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return nil, fmt.Errorf("cannot get subset ending at > size of connections (requested j: %v)", j)
	}
	return n.Connections[i:j], nil
}

func (n *Neuron) MustGetConnections(i, j int) []*Connection {
	ons, err := n.GetConnections(i, j)
	if err != nil {
		panic(err)
	}
	return ons
}

func (n *Neuron) SetConnection(i int, c *Connection) error {
	if i < 0 {
		return fmt.Errorf("cannot set Connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return fmt.Errorf("cannot set Connection at index > size of connections, %v (requested %v)", len(n.Connections), i)
	}
	n.Connections[i] = c
	return nil
}

func (n *Neuron) MustSetConnection(i int, c *Connection) {
	err := n.SetConnection(i, c)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SetConnections(i, j int, cs []*Connection) error {
	if i == j {
		return nil
	}
	if i > j {
		return fmt.Errorf("cannot set subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return fmt.Errorf("cannot set subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return fmt.Errorf("cannot set subset ending at > size of connections (requested j: %v)", j)
	}
	q := j - i
	if len(cs) != q {
		return fmt.Errorf("cannot set a subset of Connections to a set of Connections of different length. target subset length: %v, provided set length: %v", q, len(cs))
	}
	for k := 0; k < q; k++ {
		n.Connections[k+i] = cs[k]
	}
	return nil
}

func (n *Neuron) MustSetConnections(i, j int, cs []*Connection) {
	err := n.SetConnections(i, j, cs)
	if err != nil {
		panic(err)
	}
}

func (n *Neuron) SwapOutConnection(i int, c *Connection) (*Connection, error) {
	if i < 0 {
		return nil, fmt.Errorf("cannot swap connection at index < 0 (requested %v)", i)
	}
	if i >= len(n.Connections) {
		return nil, fmt.Errorf("cannot swap connection at index > size of connections, %v (requested %v)", len(n.Connections), i)
	}
	on := n.MustGetConnection(i)
	n.MustSetConnection(i, c)
	return on, nil
}

func (n *Neuron) MustSwapOutConnection(i int, c *Connection) *Connection {
	oc, err := n.SwapOutConnection(i, c)
	if err != nil {
		panic(err)
	}
	return oc
}

func (n *Neuron) SwapOutConnections(i, j int, cs []*Connection) ([]*Connection, error) {
	if i == j {
		return nil, nil
	}
	if i > j {
		return nil, fmt.Errorf("cannot swap subset [%v, %v), i must be less than or equal to j", i, j)
	}
	if i < 0 {
		return nil, fmt.Errorf("cannot swap subset starting at < 0 (requested i: %v)", i)
	}
	if j > len(n.Connections) {
		return nil, fmt.Errorf("cannot swap subset ending at > size of connections (requested j: %v)", j)
	}
	q := j - i
	if len(cs) != q {
		return nil, fmt.Errorf("cannot swap a subset of Connections with a set of Connections of different length. target subset length: %v, provided set length: %v", q, len(cs))
	}
	ols := n.MustGetConnections(i, j)
	n.MustSetConnections(i, j, cs)
	return ols, nil
}

func (n *Neuron) MustSwapOutConnections(i, j int, cs []*Connection) []*Connection {
	ons, err := n.SwapOutConnections(i, j, cs)
	if err != nil {
		panic(err)
	}
	return ons
}

//////////////////////////////////////// Connection ////////////////////////////////////////

func (c *Connection) Equals(c2 *Connection) (bool, string) {
	// NOTE(justin): To avoid misleading behavior, nil values are not considered equal.
	if c == nil || c2 == nil {
		return false, "one or both of the connections are nil"
	}

	// NOTE(justin): This introduces a lot of calculations, exponentially so as the network grows in layers.
	// This line means that Neuron.Equals calls Connection.Equals which then calls Neuron.Equals again on its connecting
	// neuron. And while this will eventually terminate, it means there will be a lot of checks.
	if equal, reason := c.To.Equals(c2.To); !equal {
		return false, reason
	}

	if c.weight != c2.weight {
		return false, "connections' weights do not match"
	}
	if c.dNetDWeight != c2.dNetDWeight {
		return false, "connections' dNetDWeights do not match"
	}
	if c.dLossDWeight != c2.dLossDWeight {
		return false, "connections' dLossDWeights do not match"
	}
	if c.dNetDPrevValue != c2.dNetDPrevValue {
		return false, "connections' dNetDPrevValues do not match"
	}

	if len(c.weightNudges) != len(c2.weightNudges) {
		return false, "connections do not have same number of weight nudges"
	}
	for wi := range c.weightNudges {
		if c.weightNudges[wi] != c2.weightNudges[wi] {
			return false, "connections' biasNudges do not match"
		}
	}

	return true, ""
}
