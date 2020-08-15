package network

import (
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"testing"
)

func Test_SerializeAndDeserializeAreInverses(t *testing.T) {
	spec := Spec{
		NeuronMap:    []int{4, 4, 4, 4},
		InputLabels:  []string{"a", "b", "c", "d"},
		OutputLabels: []string{"1", "2", "3", "4"},
	}
	nw, _ := New(spec)

	_ = nw.ForwardPass([]float64{1, 0, 0, 0}, activationfunction.Sigmoid)
	_ = nw.BackwardPass([]float64{1, 0, 0, 0})
	nw.RecordNudges()

	jt := NewJsonTranslator()
	s, _ := jt.Serialize(nw)
	nw2, _ := jt.Deserialize(s)
	s2, _ := jt.Serialize(nw2)

	if equal, reason := nw.Equals(nw2); !equal {
		t.Fatalf("original network and deserialized network do not equal each other: %v", reason)
	}

	if s != s2 {
		t.Fatalf("original JSON encoding and deserialized network JSON encoding do not equal each other")
	}
}
