# jnet

A Go library for creating, training, and using multilayer perceptron neural networks.

## Installation

`go get -u github.com/Insulince/jnet`

## Usage

This library is intended for traditional deep learning neural networks which learn via gradient descent of the mean-squared-error loss function. There is **no support for convolutional neural networks**, however I do have plans to try to add that in at some point. Some preliminary work has been done on the `cnn` branch but as of my writing nothing is usable yet.

### Structure

The core library is in the [`network`](https://github.com/Insulince/jnet/blob/master/pkg/network) package. In this package are the top-level types:

- Network - The top structure that will be most interesting to end users. Has functions for managing its layers.
- Layer - A series of neurons corresponding to one layer are tracked by this structure. Has functions for managing its neurons.
- Neuron - The base unit of a neural network. The internals of a neuron are tweaked during the training process and then used to make predictions. Each neuron is connected to every neuron in the previous layer (except input neurons, for there is no previous layer) via that neurons Connections.
- Connection - Each neuron has a set of connections to the previous layer (except input neurons, for there is no previous layer). Connections track weights and are tweaked during the training process then used to make predictions.

### Creating a Network

#### New Networks

A new network should be created via `network.From`, in which a simplified network spec is provided and a network is built up from it. While it is possible to erect a network completely from scratch yourself, it is much easier and safer to do so via `network.From`.

A network spec is composed of four fields:

- `NeuronMap` - A slice of `int`s in which the length of the slice corresponds to the number of layers in the network, and the value of the `int` at each index corresponds to the number of neurons in the layer at that index. The slice must be at least of size 2, to indicate an input and output layer, and each value in the slice must be at least 1 to indicate at least one neuron in that layer.
- `InputLabels` - A slice of `string`s which correspond index-wise to the neurons in the input layer. This is purely for organizational purposes and has no effect on network efficacy. The slice must be the same size as the number of neurons in the first layer.
- `OutputLabels` - A slice of `string`s which correspond index-wise to the neurons in the output layer. This is for organizational purposes but also the neuron with the greatest confidence when making a prediction returns its output label as well. The slice must be the same size as the number of neurons in the last layer.
- `ActivationFunctionName` - An `activationfunction.Name` (`string`) which corresponds to the activation function you want your network to utilize for non-linearization.

#### Existing Networks

Existing networks can be stored and retrieved via one of the translations supported:

- [JSON](https://github.com/Insulince/jnet/blob/master/pkg/network/json.go) - Most transportable format, however it is rather heavy and seems to grow in size exponentially with the size of the network. Most fitting for small networks that one prefers to be somewhat human readable.
- [gob](https://github.com/Insulince/jnet/blob/master/pkg/network/gob.go) - Compact format, great for storage. Exclusive to golang. Use `WtihCompression` option to get even smaller results.
- [protocol buffers](https://github.com/Insulince/jnet/blob/master/pkg/network/proto.go) - Compact format, great for storage. Can be unmarshalled into other languages if protos are generated for them via the [networks.proto](https://github.com/Insulince/jnet/blob/master/pkg/network/networks.proto) file. Use `WithCompression` option to get even smaller results.

### Operating a Network

A network can be operated manually to train it to generalize inputs, but the process is rather arduous. Nevertheless, it may be of use anyway if you require more fine-grained control of the training process than the `trainer` package provides. For information on the `trainer` package to streamline the process, see the [training section](#training-a-network).

It is useful to think of a network as a big black box. There are a series of dials and levers on the black box that can be tweaked to make a given input to the black box yield a certain output. What is special here though is that this black box has tools that allow you to reason about its state as it executes:

1. `network.Network.ForwardPass` - Computes an output from a given input and records its state as it does so.
2. `network.Network.BackwardPass` - Calculates the impact each piece of that state had on the output.
3. `network.Network.RecordNudges` - Records the nudges to each of those levers and dials which will bring the output closer to the desired output.
4. `network.Network.AdjustWeights` - Adjust those levers and dials so on the next input the output will be closer.

There are no restrictions on the order in which you execute these from a library perspective. These functions are simply available. However, the intended way to run this is to do steps 1-3 many times over across different inputs and then to do step 4 and clear the network of any lingering state (via `network.Network.ResetForPass`), then start over. This is how generalization of your inputs is to be achieved.

You can use this information to nudge the black box in the right direction towards your desired outputs across all inputs, which basically means you can have the black box adjust its dials and levers for you to achieve your desired outputs by generalizing your inputs.

The system is straightforward. Create a network, create a batch of training data, run a forward pass, backward pass, and record the nudges for each entry in the batch, after the batch is complete adjust the weights of the network, then repeat as many times as appropriate.

#### Operating Example

```go
package main

import (
	"github.com/Insulince/jnet/pkg/network"
	"github.com/Insulince/jnet/pkg/trainer"
)

const (
	iterations = 100
	learningRate = 0.1
)

var (
	// This would normally contain real training data, but is left empty for this example.
	tds trainer.Data
)

func main() {
	// Create a network.
	nw := network.MustFrom(network.Spec{/* network specifics here */})
	
	// For every desired iteration...
	for i := 0; i < iterations; i++ {
		// Create a new mini batch of random training datums.
        miniBatch := tds.MiniBatch(16)

		// Clear the network of any state from the previous iteration/batch.
		nw.ResetForBatch()
		
		// For every training datum in the batch...
		for _, td := range batch {
			// Clear the network of any state from the previous pass.
			nw.ResetForPass()
			// Push an input (td.Data) through the network.
			_ = nw.ForwardPass(td.Data)
			// Calculate the impact that each part of the network had on the output.
			_ = nw.BackwardPass(td.Truth)
			// Record the direction each part of the network needs to move to bring the actual output closer to the desired output for this training datum.
			nw.RecordNudges()
		}
		
		// After running a pass across the entire batch, actually adjust the network to improve performance on the next batch.
		nw.AdjustWeights(learningRate)
	}
	
	// After the above process completes, nw will be in a state where it has generalized the training data inputs and should perform better on new inputs than it had before.
}
```

This is the basic idea behind training a network to learn from its training data. This process is streamlined in the `trainer` package.

### Training a Network

As described in the [operating section](#operating-a-network), the `trainer` package is a package for streamlining the training process.

#### Training Data

Unsurprisingly, the most important thing you will need for training is labeled training data, and a lot of it. Training data should be built up in a `trainer.Data` structure, which is a slice of structs which contain:

- `Data` - The input to your network as a slice of `float64`s. There should be as many values in this slice as there are neurons in the input layer.
- `Truth` - The desired output from your network as a slice of `float64`s. There should be as many values in this slice as there are in the output layer.

There are no mechanisms to assist with building up this structure provided in this library, you must provide your data in this format. Be it some data massaging after reading from a file or a database, or just manually writing it in.

#### Trainer

Once you have your training data ready, you need to create a `trainer.Trainer` via `trainer.New`.

`trainer.New` accepts 3 arguments. The first is a `trainer.Configuration`, which we will go into in a moment, the second is your training data, and the last is an `io.Writer` for logging the output of the training process. Providing `nil` for the `io.Writer` results in logs being written to stdout. You can use `io.Discard` if you wish to silence the training logs.

A `trainer.Configuration` contains fields for defining how the training process should go:

- `LearningRate` - This is how aggressive the network should adjust its weights after processing a batch of training data. Too high and your network will thrash wildly in results, unable to find a local optimum. Must be greater than zero.
- `MiniBatchSize` - The size of the batch of training data that trainer will extract from your full set of training data. This value must be greater than 0 but less than or equal to the size of your training data slice.
- `AverageLossCutoff` - The training process will exit when the **average** loss (square of the difference between actual output and desired output) is less than or equal to this value. Must be greater than or equal to 0.
- `MinLossCutoff` - The training process will exit when **any** loss is less than or equal to this value. Must be greater than or equal to 0.
- `MaxIterations` - The training process will exit after this many batches are processed. Must be greater than or equal to 0.
- `Timeout` - The training process will exit after this much time has passed. Setting to `0` means there is no timeout.

The first of the exit conditions which is met will result in the training process exiting, so if `MinLossCutoff` is reached before `MaxIterations`, then the training process will exit anyway.

#### Training

Now that you have both your training data and your trainer, you are ready to actually train your network. This is done quite simply via `trainer.Train` which accepts a `network.Network` as an argument and results in an `error`. If the returned `error` is `nil` then the training process completed successfully.

All the training process does is what is shown in the [operating example](#operating-example) above but in a more friendly way to the end user and with progress logging.

#### Tips

When creating a `trainer.Configuration`, you should carefully choose the values for each of the fields. Below are some tips on ways to improve the training process.

- `LearningRate` - This is a scalar value against the weight and bias adjustments to the network. Setting this value too high, and your network will thrash wildly in results as it overcompensates for the nudges, unable to find a local optimum. Setting it to0 low and training progress will grind nearly to a halt while advancing towards a local optimum. The tradeoff is basically speed of progress for quality of progress. If you can afford to wait a long time, a lower value may give better results, but going too low is quickly computationally prohibitive. Recommended starting value is `0.1`, but adjust accordingly based on what seems to yield best results.
- `MiniBatchSize` - The size of the batch of training data that trainer will extract from your full set of training data. The tradeoff here is again speed of progress for quality of progress. If you have a very large mini batch size then your network will be better at generalizing inputs, for it is adjusting its weights based on _many_ instances of training data, but it requires more computations per batch since there are more piece of data to run passes against. Setting this too low will result in a failure to generalize and in fact over-fitting could occur, and the network will simply memorize all of your inputs. So again, too low, and you won't have any progress, but too high, and it becomes computationally prohibitive. Recommended starting value is a power of 2 less than the size of your set of training data, but not greater than 64. 32 is usually a good starting point.
- `AverageLossCutoff` - This is purely a decision you need to make. How low would you like the loss to be on average for your network for you to be comfortable that it is making the right predictions? The lower the better, but too low, and it becomes computationally prohibitive and too high  and the network is meaningless. Recommended starting value is `0.1` or 100 times greater than `MinLossCutoff`.
- `MinLossCutoff` - This is also a decision you need to make. If your network can achieve a loss this low, are you comfortable that it is making the right predictions? The lower the better, but too low, and it becomes computationally prohibitive and too high and the network is meaningless. Recommended starting value is `0.001` or 100 times less than `AverageLossCutoff`.
- `MaxIterations` - Purely depends on what you need for your network. Too low and training won't be valuable. Too high and it's computationally prohibitive. Recommended starting value is `1000` until you feel good about your network to invest some serious computation time into training it, at which point you should probably up it to `100000` or some even greater power of 10.
- `Timeout` - Depends on how much time you have to invest in the training of your network. If you don't want to train for more than a few seconds, so you can quickly test something, this would be your way of achieving that regardless of the other settings in the config. Too low and training is meaningless. Too high and it's computationally prohibitive. Recommended starting value is `0` to disable the timeout while you tweak other settings. Once happy, this is still a decision for you to make around what you need.


### Making Predictions with a Network

Once your network is trained you are ready to test it against some new data to see how it responds. This can be done via `network.Network.Predict` which accepts a `[]float64` as input (again, the slice must be the same size as the number of neurons in the input layer) and returns, in order, the `string` corresponding to the output label of the neuron with the highest value, a `float64` corresponding to the value of that same neuron, and an `error`. This only returns the highest confidence neuron information, which is effectively the network's output for this input, but if you are more interested in what the network thought about all possible outputs, instead of just the single highest confidence, you can fetch the entire last layer after making a prediction via `network.Network.LastLayer` and inspect each neuron that way.

## Example

The following example erects a simple network made of 4 layers. The first layer is the input layer with 5 neurons, and the last layer is the output layer with 3 neurons. The other two layers are hidden layers, each also containing 3 neurons. The output neurons are labeled in order as "apple", "banana", and "orange". The input neurons did not require any explicit labeling for this example, so an empty slice of the proper size is passed instead, but you can provide input labels if needed. Following initial creation we proceed to training, and the first step of that is to define some training data. Due to this being a very simple example there is only one training datum defined (and it is defined arbitrarily, mind you, this example is not intended to actually yield a meaningful result, rather it's just to show you the structure and flow of the API). In this case the provided inputs correspond to the output "orange". Following this is the configuration of how the training procedure should
go, where we set the training parameters are set. Next we run the training process, which will take up 99% of the execution time of this program. Once complete, we run a prediction and to test the efficacy of our network.

```go
package main

import (
	"fmt"
	"github.com/Insulince/jnet/pkg/trainer"

	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/Insulince/jnet/pkg/network"
	"github.com/pkg/errors"
)

func main() {
	// Create a neuron map.
	nm := []int{5, 3, 3, 3}
	// Create labels for the input neurons. Must have same length as first layer of network (5).
	ils := make([]string, nm[0])
	// Create labels for the output neurons. Must have same length as last layer of network (3).
	ols := []string{"apple", "banana", "orange"}
	// Choose an activation function for your network.
	af := activationfunction.MustGetFunction(activationfunction.NameSigmoid)

	// Create the network.
	nw, err := network.From(network.Spec{
		NeuronMap:              nm,
		InputLabels:            ils,
		OutputLabels:           ols,
		ActivationFunctionName: af,
	})
	if err != nil {
		panic(errors.Wrap(err, "network from spec"))
	}

	// Define some training data.
	td := trainer.Data{
		{
			Data:  []float64{0, 0, 1, 0, 1}, // This input...
			Truth: []float64{0, 0, 1},       // ... corresponds to this output.
		},
		// Define further training data here...
	}

	// Define some parameters for the training process.
	tc := trainer.Configuration{
		LearningRate:      0.01,
		MiniBatchSize:     1,
		AverageLossCutoff: 0.1,
		MinLossCutoff:     0.0001,
		MaxIterations:     100000,
		Timeout:           0,
	}

	// Create a network trainer.
	t := trainer.New(tc, td, nil)

	// Execute training.
	if err = t.Train(nw); err != nil {
		panic(errors.Wrap(err, "training network"))
	}

	// Test the network.
	input := []float64{0, 0, 1, 0, 1}
	prediction, confidence, err := nw.Predict(input)
	if err != nil {
		panic(errors.Wrap(err, "making prediction"))
	}
	fmt.Printf("%s (%v%% confident)\n", prediction, confidence*100)
}
```

The main limitations of this example is that we would need more training data, and a larger network to do anything worthwhile. For a better example, take a look at [cmd/simple/main.go](https://github.com/Insulince/jnet/blob/master/cmd/simple/main.go) to see how a more realistic network is trained to read a 5 by 5 pixel grid of numbers.

## Goals/TODO

- [x] Serialize and Deserialize networks to and from strings - **[translate.go](https://github.com/Insulince/jnet/blob/master/pkg/translate.go) defines interfaces for network serialization and deserialization. Supported options are [JSON](https://github.com/Insulince/jnet/blob/master/pkg/network/json.go), [gob](https://github.com/Insulince/jnet/blob/master/pkg/network/gob.go), and [protocol buffers](https://github.com/Insulince/jnet/blob/master/pkg/network/proto.go)**
- [ ] Documentation (lol)
- [x] Store and Read networks outside of program - [translate.go](https://github.com/Insulince/jnet/blob/master/pkg/) enables this
- [ ] Command Line Interface
- [x] Verbose & Silent mode - **trainer.New accepts an io.Writer. Provide io.Discard to train without output**
- [ ] Allow different activation functions per layer
- [ ] Expose statistics about network in Public API
- [ ] Concurrency/Parallelism
- [x] Stabilize Library (no panics for misconfiguration or silly mistakes)
- [ ] Standardize and export common error cases for downstream consumption
- [ ] Cancellation of training process mid-session
- [ ] Debug Relu activation function
- [ ] Implement proper error wrapping
- [ ] Makefile for tests and proto gen
- [ ] Sentinel Errors

## Inspiration and Thanks

- 3blue1brown's series on [Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- Brandon Rohrer's [How Deep Neural Networks Work](https://www.youtube.com/watch?v=ILsA4nyG7I0)
- Andrej Karpathy's lectures, specifically [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)
- Numerous Medium articles
