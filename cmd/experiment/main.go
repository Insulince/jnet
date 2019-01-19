package main

import (
	"fmt"
	"jnet/pkg/jnet"
)

func init() {
	//rand.Seed(time.Now().Unix())
}

func main() {
	nw := jnet.NewNetwork(1, 1, 1, 1, 1, 1)

	nw.Layers[1].Neurons[0].Connections[0].Weight = 0.25
	nw.Layers[1].Neurons[0].Bias = 0.0
	nw.Layers[2].Neurons[0].Connections[0].Weight = 0.75
	nw.Layers[2].Neurons[0].Bias = 0.7
	nw.Layers[3].Neurons[0].Connections[0].Weight = 1.0
	nw.Layers[3].Neurons[0].Bias = 1.0
	nw.Layers[4].Neurons[0].Connections[0].Weight = 0.1
	nw.Layers[4].Neurons[0].Bias = 0.2
	nw.Layers[5].Neurons[0].Connections[0].Weight = 0.4
	nw.Layers[5].Neurons[0].Bias = 2.0

	nw.Predict([]float64{1.0})
	nw.CalculateLoss([]float64{1.0})

	fmt.Println(nw.Layers[0].Neurons[0].Value)
	fmt.Println(nw.Layers[1].Neurons[0].Value)
	fmt.Println(nw.Layers[2].Neurons[0].Value)
	fmt.Println(nw.Layers[3].Neurons[0].Value)
	fmt.Println(nw.Layers[4].Neurons[0].Value)
	fmt.Println(nw.Layers[5].Neurons[0].Value)
	fmt.Println()
	fmt.Println(nw.Loss)
}
