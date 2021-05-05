# jnet

A Go library for creating, training, and using multilayer perceptron neural networks.

## Installation

`go get -u github.com/Insulince/jnet`

## Usage

TODO

#### Example 

The following example erects a simple network made of 4 layers. The first layer is the input layer with 5 neurons, and the last layer is the output layer with 3 neurons. The other two layers are hidden layers, each also containing 3 neurons. The output neurons are labeled in order as "apple", "banana", and "orange". The input neurons did not require any explicit labeling for this example, so an empty slice of the proper size is passed instead, but you can provide input labels if needed. Following initial creation we proceed to training, and the first step of that is to define some training data. Due to this being a very simple example there is only one training datum defined (and it is defined arbitrarily, mind you, this example is not intended to actually yield a meaningful result, rather its just to show you the structure and flow of the API). In this case the provided inputs correspond to the output "orange". Following this is the configuration of how the training procedure should go, where we set the learning rate, training iterations, minibatch size (the minibatch size is 1 because there is only 1 training datum, this is not how it would work in a real training session), and the average loss cutoff (training is stopped if the average loss is below this number). Next we run the training process, which will take up 99% of the execution time of this program. Once complete, we run a prediction and hope to see a reasonable result.

```go
package main

import (
	"fmt"
	"github.com/Insulince/jnet"
)

func main() () {
	// Create a neuron map.
	nm := []int{5, 3, 3, 3}
	// Create labels for the input neurons.
	il := make([]string, nm[0])
	// Create labels for the output neurons.
	ol := []string{"apple", "banana", "orange"}
	// Create the network.
	nw, err := jnet.NewNetwork(nm, il, ol)
	if err != nil {
		panic(err)
	}

	trainingData := jnet.TrainingData{
		{
			Data:  []float64{0, 0, 1, 0, 1},
			Truth: []float64{0, 0, 1},
		},
	}

	trainConfig := jnet.TrainingConfiguration{
		LearningRate:       0.01,
		Iterations: 100000,
		MiniBatchSize:      1,
		AverageLossCutoff:  0.1,
	}

	err = nw.Train(trainingData, trainConfig)
	if err != nil {
		panic(err)
	}

	input := []float64{0, 0, 1, 0, 1}
	fmt.Println(nw.Predict(input))
}
```

The main limitations of this example is that we would need more training data and a larger network to do anything worth while. For a better example, take a look at [cmd/train/main.go](https://github.com/Insulince/jnet/blob/master/cmd/train/main.go) to see how a more realistic network is trained to read a 5 by 5 pixel grid of numbers.

## Goals/TODO

 - [x] Serialize and Deserialize networks to and from strings
 - [ ] Documentation (lol)
 - [x] Store and Read networks ~~from file~~ outside of program.
 - [ ] Command Line Interface
 - [ ] Verbose & Silent mode
 - [ ] Allow different activation functions
 - [ ] Expose statistics about network in Public API
 - [ ] Concurrency/Parallelism
 - [x] Stabilize Library (no panics for misconfiguration or silly mistakes)
 - [ ] Standardize and export common error cases for downstream consumption
 - [ ] Cancellation of training process mid-session
 - [ ] Debug Relu activation function
 - [ ] Implement proper error wrapping
 - [ ] Makefile for tests and proto gen

## Inspiration and Thanks

- 3blue1brown's series on [Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- Brandon Rohrer's [How Deep Neural Networks Work](https://www.youtube.com/watch?v=ILsA4nyG7I0)
- Andrej Karpathy's lectures, specifically [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)
- Numerous Medium articles
