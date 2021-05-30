package network

import (
	"github.com/golang/protobuf/proto"
	"github.com/pkg/errors"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/Insulince/jnet/pkg/network/networkspb"
)

type (
	protoTranslator struct {
		opts []TranslatorOption
	}
)

var _ Translator = new(protoTranslator)

func NewProtoTranslator(opts ...TranslatorOption) Translator {
	return protoTranslator{
		opts: opts,
	}
}

func (pt protoTranslator) Serialize(nw Network) ([]byte, error) {
	pnw := toProto(nw)
	bs, err := proto.Marshal(pnw)
	if err != nil {
		return nil, errors.Wrap(err, "proto marshalling")
	}

	for i, opt := range pt.opts {
		bs, err = opt.Serialize(bs)
		if err != nil {
			return nil, errors.Wrapf(err, "translator serialize option %v", i)
		}
	}

	return bs, nil
}

// MustSerialize calls Serialize but panics if an error is encountered.
func (pt protoTranslator) MustSerialize(nw Network) []byte {
	bs, err := pt.Serialize(nw)
	if err != nil {
		panic(errors.Wrap(err, "must serialize"))
	}
	return bs
}

func (pt protoTranslator) Deserialize(bs []byte) (Network, error) {
	var err error
	for i, opt := range pt.opts {
		bs, err = opt.Deserialize(bs)
		if err != nil {
			return nil, errors.Wrapf(err, "translator deserialize option %v", i)
		}
	}

	pnw := &networkspb.Network{}
	if err := proto.Unmarshal(bs, pnw); err != nil {
		return nil, errors.Wrap(err, "proto unmarshalling")
	}
	nw, err := fromProto(pnw)
	if err != nil {
		return nil, errors.Wrap(err, "from proto")
	}
	return nw, nil
}

// MustDeserialize calls Deserialize but panics if an error is encountered.
func (pt protoTranslator) MustDeserialize(bs []byte) Network {
	nw, err := pt.Deserialize(bs)
	if err != nil {
		panic(errors.Wrap(err, "must deserialize"))
	}
	return nw
}

func toProto(nw Network) *networkspb.Network {
	pnw := &networkspb.Network{}

	// NOTE: Implicitly assumes the entire network uses the same activation
	// function AND that there exists a first layer AND there exists a first
	// neuron.
	pnw.ActivationFunctionName = string(nw[0][0].ActivationFunctionName)

	var pls []*networkspb.Layer
	for _, l := range nw {
		pl := &networkspb.Layer{}

		var pns []*networkspb.Neuron
		for _, n := range l {
			pn := &networkspb.Neuron{}

			pn.Label = n.label
			pn.Bias = n.bias

			var pcs []*networkspb.Connection
			for _, c := range n.Connections {
				pc := &networkspb.Connection{}

				pc.Weight = c.weight

				pcs = append(pcs, pc)
			}
			pn.Connections = pcs

			pns = append(pns, pn)
		}
		pl.Neurons = pns

		pls = append(pls, pl)
	}
	pnw.Layers = pls

	return pnw
}

func fromProto(pnw *networkspb.Network) (Network, error) {
	nw := Network{}

	for _, pl := range pnw.Layers {
		var l Layer

		for _, pn := range pl.Neurons {
			n := &Neuron{}

			n.label = pn.Label
			n.bias = pn.Bias
			n.ActivationFunctionName = activationfunction.Name(pnw.ActivationFunctionName)
			n.activationFunction = activationfunction.MustGetFunction(activationfunction.Name(pnw.ActivationFunctionName))

			for _, pc := range pn.Connections {
				c := &Connection{}

				c.weight = pc.Weight

				n.Connections = append(n.Connections, c)
			}

			l = append(l, n)
		}

		nw = append(nw, l)
	}

	// NOTE: Must reconnect all neurons in network using existing connections so
	// that a linked list of neurons is successfully built.
	if err := nw.ReconnectNeurons(); err != nil {
		return nil, errors.Wrap(err, "reconnecting neurons")
	}

	return nw, nil
}
