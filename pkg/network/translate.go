package network

type Serializer interface {
	Serialize(Network) (string, error)
	MustSerialize(Network) string
}

type Deserializer interface {
	Deserialize(string) (Network, error)
	MustDeserialize(string) Network
}

type Translator interface {
	Serializer
	Deserializer
}
