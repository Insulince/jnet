package main

import (
	"fmt"
	"jnet/pkg/jnet"
)

func init() {
	//rand.Seed(time.Now().Unix())
}

func main() {
	nw := jnet.NewNetwork(1, 1, 1, 1)
	nw.LearningRate = 0.01
	nw.TrainingIterations = 50
	nw.MiniBatchSize = 3

	nw.Predict([]float64{1.0})
	nw.CalculateLoss([]float64{1.0})

	fmt.Println(nw.Layers[0].Neurons[0].Value)
	fmt.Println(nw.Layers[1].Neurons[0].Value)
	fmt.Println(nw.Layers[2].Neurons[0].Value)
	fmt.Println(nw.Layers[3].Neurons[0].Value)
	fmt.Println()
	fmt.Println(nw.Loss)
}
