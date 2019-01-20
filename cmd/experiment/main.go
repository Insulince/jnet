package main

import (
	"fmt"
	"jnet"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func main() {
	nw := jnet.NewNetwork(25, 16, 16, 10)

	nw.Train(jnet.FiveByFiveTrainingData, 0.1, 2500000, 32)

	hd := jnet.HumanData{
		Data: [][]float64{
			{1, 1, 1, 0, 0},
			{1, 0, 1, 0, 0},
			{1, 1, 1, 0, 0},
			{0, 0, 1, 0, 0},
			{1, 1, 1, 0, 0},
		},
	}
	realData := hd.ToTrainingData().Data

	nw.Predict(realData)

	fmt.Println(nw.GetResults())
	fmt.Println()
	fmt.Println(nw.Serialize())
	fmt.Println()
}
