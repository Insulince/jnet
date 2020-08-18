package network

import (
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"testing"
)

func Test_SerializeAndDeserializeAreInverses(t *testing.T) {
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

	jt := NewJsonTranslator()
	s := jt.MustSerialize(nw)
	nw2 := jt.MustDeserialize(s)
	s2 := jt.MustSerialize(nw2)

	if equal, reason := nw.Equals(nw2); !equal {
		t.Fatalf("original network and deserialized network do not equal each other: %v", reason)
	}

	if s != s2 {
		t.Fatalf("original JSON encoding and deserialized network JSON encoding do not equal each other")
	}
}
