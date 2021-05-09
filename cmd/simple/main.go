// Program simple show cases a simple and contrived example of training a neural network to recognize data. I don't
// think its fair to say that this program shows learning or pattern recognition, because the network generated is
// extremely overly fitted to the data, since the data covers every possible input, but that is the nature of working
// with such a contrived example as this.
// Now for the example details. This network takes in a 5x5 grid of input variables which contain 7 segment display
// style numbers encoded in them. The output of this network is intended to classify each input as a single digit
// number.
// There are only 32 possible ways to represent these seven segment display encodings, and all 32 are used here, which
// is why this example is over fitted. This concept does scale up, however. If we were to make this a 200x200 grid of
// inputs, and loaded in examples of handwritten numbers, learning and pattern recognition could certainly arise from
// the proper training set.
package main

import (
	"fmt"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/Insulince/jnet/pkg/network"
	"github.com/Insulince/jnet/pkg/trainer"
	"log"
	"math/rand"
	"os"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func main() {
	trainingData := trainer.Data{
		// 0
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
			},
			Truth: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 1, 1, 0,
			},
			Truth: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 1, 1,
			},
			Truth: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		// 1
		{
			Data: []float64{
				1, 0, 0, 0, 0,
				1, 0, 0, 0, 0,
				1, 0, 0, 0, 0,
				1, 0, 0, 0, 0,
				1, 0, 0, 0, 0,
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 0, 0, 0,
				0, 1, 0, 0, 0,
				0, 1, 0, 0, 0,
				0, 1, 0, 0, 0,
				0, 1, 0, 0, 0,
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
			},
			Truth: []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		// 2
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				0, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
				1, 0, 0, 0, 0,
				1, 1, 1, 0, 0,
			},
			Truth: []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 0, 0, 1, 0,
				0, 1, 1, 1, 0,
				0, 1, 0, 0, 0,
				0, 1, 1, 1, 0,
			},
			Truth: []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 0, 0, 1,
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 0,
				0, 0, 1, 1, 1,
			},
			Truth: []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		},
		// 3
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				0, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
				0, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
			},
			Truth: []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 0, 0, 1, 0,
				0, 1, 1, 1, 0,
				0, 0, 0, 1, 0,
				0, 1, 1, 1, 0,
			},
			Truth: []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 0, 0, 1,
				0, 0, 1, 1, 1,
				0, 0, 0, 0, 1,
				0, 0, 1, 1, 1,
			},
			Truth: []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		},
		// 4
		{
			Data: []float64{
				1, 0, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
			},
			Truth: []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 0, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 1, 1, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
			},
			Truth: []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 0, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 1, 1,
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
			},
			Truth: []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		},
		// 5
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				1, 0, 0, 0, 0,
				1, 1, 1, 0, 0,
				0, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 1, 0, 0, 0,
				0, 1, 1, 1, 0,
				0, 0, 0, 1, 0,
				0, 1, 1, 1, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 0,
				0, 0, 1, 1, 1,
				0, 0, 0, 0, 1,
				0, 0, 1, 1, 1,
			},
			Truth: []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		},
		// 6
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				1, 0, 0, 0, 0,
				1, 1, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 1, 0, 0, 0,
				0, 1, 1, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 1, 1, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 0,
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 1, 1,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		},
		// 7
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 1, 0, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 1, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
				0, 0, 0, 0, 1,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		},
		// 8
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 1, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 1, 1, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 1, 1,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		},
		// 9
		{
			Data: []float64{
				1, 1, 1, 0, 0,
				1, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
				0, 0, 1, 0, 0,
				1, 1, 1, 0, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
		},
		{
			Data: []float64{
				0, 1, 1, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 1, 1, 0,
				0, 0, 0, 1, 0,
				0, 1, 1, 1, 0,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
		},
		{
			Data: []float64{
				0, 0, 1, 1, 1,
				0, 0, 1, 0, 1,
				0, 0, 1, 1, 1,
				0, 0, 0, 0, 1,
				0, 0, 1, 1, 1,
			},
			Truth: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
		},
	}

	nw, err := network.From(network.Spec{
		NeuronMap:              []int{25, 16, 16, 10},
		OutputLabels:           []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
		ActivationFunctionName: activationfunction.NameSigmoid,
	})
	if err != nil {
		log.Fatalln(err)
	}

	trainConfig := trainer.Configuration{
		LearningRate:      0.1,
		MiniBatchSize:     32,
		MaxIterations:     2500000,
		AverageLossCutoff: 0.5,
		Timeout:           1 * time.Minute,
	}

	t := trainer.New(trainConfig, trainingData, os.Stdout)

	err = t.Train(nw)
	if err != nil {
		if err != trainer.ErrTimedOut {
			log.Fatalln(err)
		}
		log.Println(err.Error(), "continuing to prediction from current network state")
	}

	// This input represents a 7.
	input := []float64{
		1, 1, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
	}

	fmt.Println("Prediction (should be 7):")
	prediction, value, err := nw.Predict(input)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println(prediction, value)
}
