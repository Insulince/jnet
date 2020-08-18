package main

import (
	"fmt"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/Insulince/jnet/pkg/network"
	"github.com/Insulince/jnet/pkg/trainer"
	"log"
	"math/rand"
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

	// TODO(justin): Make this less overly fitted
	trainConfig := trainer.Configuration{
		LearningRate:      0.1,
		Iterations:        2500000,
		MiniBatchSize:     32,
		AverageLossCutoff: 0.5,
		Timeout:           1 * time.Minute,
	}

	t := trainer.New(trainConfig, trainingData)

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
	prediction, err := nw.Predict(input)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println(prediction)
}
