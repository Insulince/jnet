# JNet

TODO

## Installation

`go get -u github.com/Insulince/jnet`

## Usage

TODO

#### Example 

The following example erects a simple network made of 4 layers. The first layer is the input layer with 5 neurons, and the last layer is the output layer with 3 neurons. The other two layers are hidden layers, each also containing 3 neurons. Each output neuron is labeled in order as "apple", "banana", and "orange". Following initial creation we proceed to training, and the first step of that is to define some training data. Due to this being a very simple example there is only one training datum defined (and it is defined arbitrarilly, mind you, this example is not intended to actually yield a meaningful result, rather just to show you the structure and flow of the API). In this case the provided inputs correspond to the output "orange". Following this is the configuration of how the training procedure should go, where we set the learning rate, training iterations, minibatch size (is also 1 because there is only 1 training datum, this is not how it would work in a real training session), and the average loss cutoff (training is stopped if the average loss is below this number). Next we run the training process, which will take up 99% of the execution time of this program. Once complete, we run a prediction and hope to see a reasonable result.

```go
package main

import (
	"fmt"
	"github.com/Insulince/jnet"
)

func main() () {
	neurons := []int{5, 3, 3, 3}
	labels := []string{"apple", "banana", "orange"}
	nw := jnet.NewNetwork(neurons, labels)

	trainingData := jnet.TrainingData{
		{
			Data:  []float64{0, 0, 1, 0, 1},
			Truth: []float64{0, 0, 1},
		},
	}

	trainConfig := jnet.TrainingConfiguration{
		LearningRate:       0.01,
		TrainingIterations: 100000,
		MiniBatchSize:      1,
		AverageLossCutoff:  0.1,
	}

	nw.Train(trainingData, trainConfig)

	input := []float64{0, 0, 1, 0, 1}
	fmt.Println(nw.Predict(input))
}
```

The main limitations of this example is that we would need more training data and a larger network to do anything worth while. For a better example, take a look at [cmd/train/main.go](https://github.com/Insulince/jnet/blob/master/cmd/train/main.go) to see how a more realistic network is trained to read a 5 by 5 pixel grid of numbers.

## Goals

TODO
