package network

import (
	"bytes"
	"encoding/gob"
	"github.com/Insulince/jnet/pkg/network/networkspb"
	"github.com/pkg/errors"
)

func init() {
	gob.Register(&networkspb.Network{})
}

type gobTranslator struct {
	opts []TranslatorOption
}

var _ Translator = new(gobTranslator)

func NewGobTranslator(opts ...TranslatorOption) Translator {
	return gobTranslator{
		opts: opts,
	}
}

func (gt gobTranslator) Serialize(nw Network) ([]byte, error) {
	var b bytes.Buffer
	err := gob.NewEncoder(&b).Encode(nw)
	if err != nil {
		return nil, errors.Wrap(err, "gob marshalling")
	}
	bs := b.Bytes()

	for i, opt := range gt.opts {
		bs, err = opt.Serialize(bs)
		if err != nil {
			return nil, errors.Wrapf(err, "translator serialize option %v", i)
		}
	}

	return bs, nil
}

func (gt gobTranslator) MustSerialize(nw Network) []byte {
	bs, err := gt.Serialize(nw)
	if err != nil {
		panic(errors.Wrap(err, "must serialize"))
	}
	return bs
}

func (gt gobTranslator) Deserialize(bs []byte) (Network, error) {
	var err error
	for i, opt := range gt.opts {
		bs, err = opt.Deserialize(bs)
		if err != nil {
			return nil, errors.Wrapf(err, "translator deserialize option %v", i)
		}
	}

	var nw Network
	if err := gob.NewDecoder(bytes.NewBuffer(bs)).Decode(&nw); err != nil {
		return Network{}, errors.Wrap(err, "gob unmarshalling")
	}

	return nw, nil
}

func (gt gobTranslator) MustDeserialize(bs []byte) Network {
	nw, err := gt.Deserialize(bs)
	if err != nil {
		panic(errors.Wrap(err, "must deserialize"))
	}
	return nw
}
