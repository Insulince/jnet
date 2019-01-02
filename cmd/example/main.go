package main

import (
	"fmt"
	"jnet/pkg/layer"
	"jnet/pkg/network"
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
	inputs := [4]float64{-1, 1, 1, -1}

	nw := network.NewNetwork()

	// Input Layer
	il := *layer.NewNilLayer(4)
	il[0].Value = inputs[0]
	il[1].Value = inputs[1]
	il[2].Value = inputs[2]
	il[3].Value = inputs[3]
	nw.CreateConnections(il)

	// Hidden Layer 1
	hl1 := *layer.NewSigmoidLayer(4)
	nw.CreateConnections(hl1).
		SetConnectionWeight(0, 0, 1).
		SetConnectionWeight(0, 2, 1).
		SetConnectionWeight(1, 1, 1).
		SetConnectionWeight(1, 3, 1).
		SetConnectionWeight(2, 1, 1).
		SetConnectionWeight(2, 3, -1).
		SetConnectionWeight(3, 0, 1).
		SetConnectionWeight(3, 2, -1)

	// Hidden Layer 2
	hl2 := *layer.NewSigmoidLayer(4)
	nw.CreateConnections(hl2).
		SetConnectionWeight(0, 0, 1).
		SetConnectionWeight(0, 1, -1).
		SetConnectionWeight(1, 0, 1).
		SetConnectionWeight(1, 1, 1).
		SetConnectionWeight(2, 2, 1).
		SetConnectionWeight(2, 3, 1).
		SetConnectionWeight(3, 2, -1).
		SetConnectionWeight(3, 3, 1)

	// Hidden Layer 3
	hl3 := *layer.NewRectifiedLinearUnitLayer(8)
	nw.CreateConnections(hl3).
		SetConnectionWeight(0, 0, 1).
		SetConnectionWeight(0, 1, -1).
		SetConnectionWeight(1, 2, 1).
		SetConnectionWeight(1, 3, -1).
		SetConnectionWeight(2, 4, 1).
		SetConnectionWeight(2, 5, -1).
		SetConnectionWeight(3, 6, 1).
		SetConnectionWeight(3, 7, -1)

	// Output Layer
	ol := *layer.NewPositiveBinaryLayer(4)
	nw.CreateConnections(ol).
		SetOutputNeuronResult(0, "Solid").
		SetOutputNeuronResult(1, "Vertical").
		SetOutputNeuronResult(2, "Diagonal").
		SetOutputNeuronResult(3, "Horizontal").
		SetConnectionWeight(0, 0, 1).
		SetConnectionWeight(1, 0, 1).
		SetConnectionWeight(2, 1, 1).
		SetConnectionWeight(3, 1, 1).
		SetConnectionWeight(4, 2, 1).
		SetConnectionWeight(5, 2, 1).
		SetConnectionWeight(6, 3, 1).
		SetConnectionWeight(7, 3, 1)

	result := nw.Process()

	nw.Log(fmt.Sprintf("Result: %v", result))
}
