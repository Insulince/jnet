package network_test

import (
	"errors"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/Insulince/jnet/pkg/network"
	"github.com/Insulince/jnet/pkg/trainer"
	"math/rand"
	"testing"
	"time"
)

var timeout = errors.New("timeout")

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Test_NetworkConverges is a sanity-check test intended to be heavily utilized during development to ensure the
// algorithm has not broken during any changes. The idea is, this test case encapsulates a network known to converge to
// a low loss level very quickly, so if this test takes longer than some pre-defined cutoff, this test will fail and
// thus we know the algorithm is broken.
// The choice to seed the randomness generator differently everytime is tricky, for it introduces some uncertainty to
// how sure we can be that this test isn't going to generate a false negative. On the one hand, introducing randomness
// makes it such that we aren't exclusively testing the mechanisms of the algorithm, we are also testing that the random
// weights are favorable to quickly converging network, hence the uncertainty. However, on the other hand the vast
// majority of randomized weights should behave imperceptibly similar to other randomized weights, so in this regard it
// expands the veracity of the test. And since it is exceedingly rare to randomly encounter a set of weights which cause
// a false negative, then to truly be sure in the event of a failing test that it isn't a false negative, simply run it
// again, or even multiple times.
func Test_NetworkConverges(t *testing.T) {
	exit := make(chan error)

	go func() {
		spec := network.Spec{
			NeuronMap:              []int{4, 4, 4, 4},
			OutputLabels:           []string{"1", "2", "3", "4"},
			ActivationFunctionName: activationfunction.NameSigmoid,
		}
		nw := network.MustFrom(spec)

		td := trainer.Data{
			{
				Data:  []float64{1, 0, 0, 0},
				Truth: []float64{1, 0, 0, 0},
			},
			{
				Data:  []float64{0, 1, 0, 0},
				Truth: []float64{1, 0, 0, 0},
			},
			{
				Data:  []float64{0, 0, 1, 0},
				Truth: []float64{0, 1, 0, 0},
			},
			{
				Data:  []float64{0, 0, 0, 1},
				Truth: []float64{0, 1, 0, 0},
			},
		}

		tc := trainer.Configuration{
			LearningRate:      0.1,
			MiniBatchSize:     len(td),
			MaxIterations:     2500000,
			AverageLossCutoff: 0.1,
		}

		tr := trainer.New(tc, td, nil)

		err := tr.Train(nw)

		exit <- err
	}()

	go func() {
		time.Sleep(1 * time.Second) // This network should converge in well under a second.
		exit <- timeout
	}()

	select {
	case err := <-exit:
		if err == timeout {
			t.Fatal("network took to long to converge")
			return
		}
		if err != nil {
			t.Fatalf("something went wrong: %v", err)
		}
	}
}
