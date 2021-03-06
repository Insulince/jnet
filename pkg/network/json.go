package network

import (
	"encoding/json"

	"github.com/pkg/errors"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
)

type jsonTranslator struct {
	opts []TranslatorOption
}

var _ Translator = new(jsonTranslator)

func NewJsonTranslator(opts ...TranslatorOption) Translator {
	return jsonTranslator{
		opts: opts,
	}
}

func (jt jsonTranslator) Serialize(nw Network) ([]byte, error) {
	bs, err := json.Marshal(nw)
	if err != nil {
		return nil, errors.Wrap(err, "json marshalling")
	}

	for i, opt := range jt.opts {
		bs, err = opt.Serialize(bs)
		if err != nil {
			return nil, errors.Wrapf(err, "translator serialize option %v", i)
		}
	}

	return bs, nil
}

// MustSerialize calls Serialize but panics if an error is encountered.
func (jt jsonTranslator) MustSerialize(nw Network) []byte {
	bs, err := jt.Serialize(nw)
	if err != nil {
		panic(errors.Wrap(err, "must serialize"))
	}
	return bs
}

func (jt jsonTranslator) Deserialize(bs []byte) (Network, error) {
	var err error
	for i, opt := range jt.opts {
		bs, err = opt.Deserialize(bs)
		if err != nil {
			return nil, errors.Wrapf(err, "translator deserialize option %v", i)
		}
	}

	var nw Network
	if err := json.Unmarshal(bs, &nw); err != nil {
		return Network{}, errors.Wrap(err, "json unmarshalling")
	}
	return nw, nil
}

// MustDeserialize calls Deserialize but panics if an error is encountered.
func (jt jsonTranslator) MustDeserialize(bs []byte) Network {
	nw, err := jt.Deserialize(bs)
	if err != nil {
		panic(errors.Wrap(err, "must deserialize"))
	}
	return nw
}

func (nw *Network) UnmarshalJSON(data []byte) error {
	var nnw Network

	// NOTE(justin): First hacky workaround: Unmarshal into a slice of layers as
	// opposed to a network to prevent an infinite recursion. If you try to
	// unmarshal into a network, this same function will be indirectly called
	// again.
	var layers []Layer
	err := json.Unmarshal(data, &layers)
	if err != nil {
		return err
	}

	// NOTE(justin): Second hacky workaround: Because networks are slices,
	// theres no fields on them to allow for pass-by-reference changes to
	// persist to the caller. This is important because if a nil network, or
	// just a network with lesser size than the json is encountered, we get an
	// index out of bounds error when directly assigning to it:
	//    nw[i] = layers[i] // Causes error if i >= len(nw) (indicating layers
	//      is larger than nw, which SHOULD be fine)
	// And we can't just append to get around this, because those changes aren't
	// persisted to the caller:
	//    nw = append(nw, layers[i]) // Doesn't persist to caller
	// So the only way to achieve a "safe" but sufficiently arbitrary
	// unmarshalling process is to use a pointer receiver and append to what it
	// points to (note that this process has been extracted into a new variable,
	// nnw, and the pointer is overwritten at the end):
	for _, layer := range layers {
		nnw = append(nnw, layer)
	}

	// NOTE(justin): Third hacky work around: This one is likely unavoidable.
	// When storing a network as a json string, the json will get extremely
	// redundant, and exponentially so as the number of layers increases,
	// because neuron's connections are effectively a linked list. So if you
	// look at the JSON you would see:
	// - A neuron in layer 1 with no connections
	// - A neuron in layer 2 with a connection pointing to the same neuron in
	//     layer 1 and all its details
	// - A neuron in layer 3 with a connection pointing to the same neuron in
	//     layer 2 with all its details including the connection that points to
	//     layer 1 with all its details
	// - etc.
	// This is not desirable and to keep the JSON small the neuron portion of
	// the connection is ignored via `json:"-"`.
	// However when we unmarshal, we want to get this relationship back, so we
	// need to hook everything back up which is done via Layer.ConnectNeurons
	// here.
	//
	// For every layer starting from the last EXCEPT the first...
	for li := len(nnw) - 1; li > 0; li-- {
		// Connect the neurons in this layer to the neurons in the previous
		// layer
		err := nnw[li].ConnectNeurons(nnw[li-1])
		if err != nil {
			return err
		}
	}

	// Overwrite the pointer with the updated network to persist the changes to
	// the caller.
	*nw = nnw

	return nil
}

// jsonNeuron is private struct for mapping all fields of a Neuron to exported
// fields so that they may be exposed in a json body by the JSON marshaller.
type jsonNeuron struct {
	Connections            []*Connection           `json:"connections"`
	ActivationFunctionName activationfunction.Name `json:"activationFunctionName"`
	Label                  string                  `json:"label"`
	Value                  float64                 `json:"value"`
	WSum                   float64                 `json:"wSum"`
	Bias                   float64                 `json:"bias"`
	DLossDValue            float64                 `json:"dLossDValue"`
	DLossDBias             float64                 `json:"dLossDBias"`
	DValueDNet             float64                 `json:"dValueDNet"`
	DNetDBias              float64                 `json:"dNetDBias"`
	BiasNudges             []float64               `json:"biasNudges"`
}

func (n *Neuron) MarshalJSON() ([]byte, error) {
	j, err := json.Marshal(jsonNeuron{
		Connections:            n.Connections,
		ActivationFunctionName: n.ActivationFunctionName,
		Label:                  n.label,
		Value:                  n.value,
		WSum:                   n.wSum,
		Bias:                   n.bias,
		DLossDValue:            n.dLossDValue,
		DLossDBias:             n.dLossDBias,
		DValueDNet:             n.dValueDNet,
		DNetDBias:              n.dNetDBias,
		BiasNudges:             n.biasNudges,
	})
	if err != nil {
		return nil, err
	}
	return j, nil
}

func (n *Neuron) UnmarshalJSON(j []byte) error {
	var t jsonNeuron
	err := json.Unmarshal(j, &t)
	if err != nil {
		return err
	}

	n.Connections = t.Connections
	n.ActivationFunctionName = t.ActivationFunctionName
	n.label = t.Label
	n.value = t.Value
	n.wSum = t.WSum
	n.bias = t.Bias
	n.dLossDValue = t.DLossDValue
	n.dLossDBias = t.DLossDBias
	n.dValueDNet = t.DValueDNet
	n.dNetDBias = t.DNetDBias
	n.biasNudges = t.BiasNudges

	af, err := activationfunction.GetFunction(n.ActivationFunctionName)
	if err != nil {
		return err
	}
	n.activationFunction = af

	return nil
}

// jsonConnection is private struct for mapping all fields of a Connection to
// exported fields so that they may be exposed in a json body by the JSON
// marshaller.
type jsonConnection struct {
	To             *Neuron   `json:"-"`
	Weight         float64   `json:"weight"`
	DNetDWeight    float64   `json:"dNetDWeight"`
	DLossDWeight   float64   `json:"dLossDWeight"`
	DNetDPrevValue float64   `json:"dNetDPrevValue"`
	WeightNudges   []float64 `json:"weightNudges"`
}

func (c *Connection) MarshalJSON() ([]byte, error) {
	j, err := json.Marshal(jsonConnection{
		To:             c.To,
		Weight:         c.weight,
		DNetDWeight:    c.dNetDWeight,
		DLossDWeight:   c.dLossDWeight,
		DNetDPrevValue: c.dNetDPrevValue,
		WeightNudges:   c.weightNudges,
	})
	if err != nil {
		return nil, err
	}
	return j, nil
}

func (c *Connection) UnmarshalJSON(j []byte) error {
	var t jsonConnection
	err := json.Unmarshal(j, &t)
	if err != nil {
		return err
	}

	c.To = t.To
	c.weight = t.Weight
	c.dNetDWeight = t.DNetDWeight
	c.dLossDWeight = t.DLossDWeight
	c.dNetDPrevValue = t.DNetDPrevValue
	c.weightNudges = t.WeightNudges

	return nil
}
