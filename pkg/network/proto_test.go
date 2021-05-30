package network

import (
	"testing"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
)

func Test_proto_SerializeAndDeserializeAreInverses(t *testing.T) {
	spec := Spec{
		NeuronMap:              []int{5, 8, 3},
		InputLabels:            []string{"0", "1", "2", "3", "4"},
		OutputLabels:           []string{"0", "1", "2"},
		ActivationFunctionName: activationfunction.NameSigmoid,
	}
	nw := MustFrom(spec)

	// NOTE: This is not included like it is for the JSON translator because
	// proto translation does not encode things like weight and bias nudges.
	// nw.MustForwardPass([]float64{1, 0, 0, 0, 0})
	// nw.MustBackwardPass([]float64{1, 0, 0})
	// nw.RecordNudges()

	pt := NewProtoTranslator(WithCompression())
	s := pt.MustSerialize(nw)
	nw2 := pt.MustDeserialize(s)
	s2 := pt.MustSerialize(nw2)

	if err := nw.Equals(nw2); err != nil {
		t.Fatalf("original network and deserialized network do not equal each other: %v", err)
	}

	if string(s) != string(s2) {
		t.Fatalf("original proto encoding and deserialized network proto encoding do not equal each other")
	}
}
