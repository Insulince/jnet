package network

import (
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"testing"
)

func Test_proto_SerializeAndDeserializeAreInverses(t *testing.T) {
	spec := Spec{
		NeuronMap:              []int{4, 4, 4, 4},
		InputLabels:            []string{"a", "b", "c", "d"},
		OutputLabels:           []string{"1", "2", "3", "4"},
		ActivationFunctionName: activationfunction.NameSigmoid,
	}
	nw := MustFrom(spec)

	nw.MustForwardPass([]float64{1, 0, 0, 0})
	nw.MustBackwardPass([]float64{1, 0, 0, 0})
	nw.RecordNudges()

	pt := NewProtoTranslator(WithCompression())
	s := pt.MustSerialize(nw)
	nw2 := pt.MustDeserialize(s)
	s2 := pt.MustSerialize(nw2)

	if err := nw.Equals(nw2); err != nil {
		// TODO: Broken likely because proto encodings don't encode everything about the network like JSON encodings do, but the `Equals` method doesn't know that and checks everything anyway.
		t.Fatalf("original network and deserialized network do not equal each other: %v", err)
	}

	if string(s) != string(s2) {
		t.Fatalf("original proto encoding and deserialized network proto encoding do not equal each other")
	}
}
