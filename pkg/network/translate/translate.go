package translate

import (
	"github.com/Insulince/jnet/pkg/network"
)

type Serializer interface {
	Serialize(network.Network) (string, error)
	MustSerialize(network.Network) string
}

type Deserializer interface {
	Deserialize(string) (network.Network, error)
	MustDeserialize(string) network.Network
}

type Translator interface {
	Serializer
	Deserializer
}
