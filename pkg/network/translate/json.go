package translate

import (
	"encoding/json"
	"fmt"
	"github.com/Insulince/jnet/pkg/network"
)

type jsonTranslator struct {
}

var _ Translator = new(jsonTranslator)

func NewJsonTranslator() Translator {
	return jsonTranslator{}
}

func (jt jsonTranslator) Serialize(n network.Network) (string, error) {
	raw, err := json.Marshal(n)
	if err != nil {
		return "", fmt.Errorf("json marshal: %w", err)
	}
	return string(raw), nil
}

func (jt jsonTranslator) MustSerialize(n network.Network) string {
	raw, err := json.Marshal(n)
	if err != nil {
		panic(fmt.Errorf("json marshal: %w", err))
	}
	return string(raw)
}

func (jt jsonTranslator) Deserialize(s string) (network.Network, error) {
	var n network.Network
	err := json.Unmarshal([]byte(s), &n)
	if err != nil {
		return network.Network{}, fmt.Errorf("json unmarshal: %w", err)
	}
	return n, nil
}

func (jt jsonTranslator) MustDeserialize(s string) network.Network {
	var n network.Network
	err := json.Unmarshal([]byte(s), &n)
	if err != nil {
		panic(fmt.Errorf("json unmarshal: %w", err))
	}
	return n
}
