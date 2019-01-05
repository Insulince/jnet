package main

import (
	"fmt"
	"jnet/pkg/connection"
	"jnet/pkg/data"
	"jnet/pkg/layer"
	"jnet/pkg/network"
	"jnet/pkg/neuron"
)

// This network is an example that was constructed based on Brandon Rohrer's excellent talk
// on Neural Networks:  https://www.youtube.com/watch?v=ILsA4nyG7I0
// Explanation:
// Construct an array, composed of 4 values. Each value can be either -1 or 1.
// Consider this array to be rolled up such that it makes a box resembling:
// +---+---+
// | 0 | 1 |
// +---+---+
// | 3 | 2 |
// +---+---+
// Where each index is mapped to its corresponding cell in this box.
// This network will identify if a configuration of -1's and 1's is either:
// - Solid
// - Horizontal
// - Vertical
// - Diagonal
// Example box:
// +----+---+
// | -1 | 1 |
// +----+---+
// | -1 | 1 |
// +----+---+
// Is created from the input array [-1, 1, 1, -1], and will output "Vertical".
//
// Note that the configurations require all input arrays to either contain all 4 values to be the same, or there to be
// 2 of 1 value and 2 of another, anything in between will have undefined behavior.
//
// This network comes PRE-TRAINED. It does not use any data science or calculus to create its determinations.
// This is just an exercise in the machinery behind neural networks.
func main() {
	nw := network.New([]layer.Layer{
		*layer.New(4, neuron.TypeInput),
		*layer.New(4, neuron.TypeSigmoid),
		*layer.New(4, neuron.TypeSigmoid),
		*layer.New(8, neuron.TypeRectifiedLinearUnit),
		*layer.New(4, neuron.TypeOutput).
			SetOutputNeuronResults(
				[]string{"Solid", "Vertical", "Diagonal", "Horizontal"},
			),
	}).SetAllConnectionWeightsTo(0.0).
		ApplyConnectionMaps([]connection.Map{
			{
				{0.0, 0.0, 1.0},
				{0.0, 2.0, 1.0},
				{1.0, 1.0, 1.0},
				{1.0, 3.0, 1.0},
				{2.0, 1.0, 1.0},
				{2.0, 3.0, -1.0},
				{3.0, 0.0, 1.0},
				{3.0, 2.0, -1.0},
			},
			{
				{0.0, 0.0, 1.0},
				{0.0, 1.0, -1.0},
				{1.0, 0.0, 1.0},
				{1.0, 1.0, 1.0},
				{2.0, 2.0, 1.0},
				{2.0, 3.0, 1.0},
				{3.0, 2.0, -1.0},
				{3.0, 3.0, 1.0},
			},
			{
				{0.0, 0.0, 1.0},
				{0.0, 1.0, -1.0},
				{1.0, 2.0, 1.0},
				{1.0, 3.0, -1.0},
				{2.0, 4.0, 1.0},
				{2.0, 5.0, -1.0},
				{3.0, 6.0, 1.0},
				{3.0, 7.0, -1.0},
			},
			{
				{0.0, 0.0, 1.0},
				{1.0, 0.0, 1.0},
				{2.0, 1.0, 1.0},
				{3.0, 1.0, 1.0},
				{4.0, 2.0, 1.0},
				{5.0, 2.0, 1.0},
				{6.0, 3.0, 1.0},
				{7.0, 3.0, 1.0},
			},
		})

	nw.Predict(data.D{1.0, -1.0, 1.0, -1.0})

	fmt.Println(nw.GetResults())
}
