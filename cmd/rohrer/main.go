// Program rohrer is a manually constructed neural network which represents the example created by Brandon Rohrer in his
// excellent introduction to Deep Neural Networks found here: https://youtu.be/ILsA4nyG7I0?t=1371
// Of course, this is an extremely contrived neural network used only for educational purposes, but the theory behind it
// should hold true for jnet as well, which it does.
// Each of the possible outcomes have input scenarios defined at the end of the main function. Simply uncomment one
// and comment out the previous one to test the different scenarios.
package main

import (
	"fmt"
	activationfunction "github.com/Insulince/jnet/pkg/activation-function"
	"github.com/Insulince/jnet/pkg/network"
	"log"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func main() {
	spec := network.Spec{
		NeuronMap:              []int{4, 4, 4, 8, 4},
		OutputLabels:           []string{"solid", "vertical", "diagonal", "horizontal"},
		ActivationFunctionName: activationfunction.NameSigmoid,
	}
	nw, err := network.From(spec)
	if err != nil {
		log.Fatalln(err)
	}

	nw.SetNeuronBiasesTo(0)
	nw.SetNeuronValuesTo(0)
	nw.SetConnectionWeightsTo(0)

	nw[3].MustSetNeuronActivationFunctionsTo(activationfunction.NameRelu)

	nw[1][0].MustSetConnectionWeights([]float64{1, 0, 0, 1})
	nw[1][1].MustSetConnectionWeights([]float64{0, 1, 1, 0})
	nw[1][2].MustSetConnectionWeights([]float64{1, 0, 0, -1})
	nw[1][3].MustSetConnectionWeights([]float64{0, 1, -1, 0})

	nw[2][0].MustSetConnectionWeights([]float64{1, 1, 0, 0})
	nw[2][1].MustSetConnectionWeights([]float64{-1, 1, 0, 0})
	nw[2][2].MustSetConnectionWeights([]float64{0, 0, 1, -1})
	nw[2][3].MustSetConnectionWeights([]float64{0, 0, 1, 1})

	nw[3][0].MustSetConnectionWeights([]float64{1, 0, 0, 0})
	nw[3][1].MustSetConnectionWeights([]float64{-1, 0, 0, 0})
	nw[3][2].MustSetConnectionWeights([]float64{0, 1, 0, 0})
	nw[3][3].MustSetConnectionWeights([]float64{0, -1, 0, 0})
	nw[3][4].MustSetConnectionWeights([]float64{0, 0, 1, 0})
	nw[3][5].MustSetConnectionWeights([]float64{0, 0, -1, 0})
	nw[3][6].MustSetConnectionWeights([]float64{0, 0, 0, 1})
	nw[3][7].MustSetConnectionWeights([]float64{0, 0, 0, -1})

	nw[4][0].MustSetConnectionWeights([]float64{1, 1, 0, 0, 0, 0, 0, 0})
	nw[4][1].MustSetConnectionWeights([]float64{0, 0, 1, 1, 0, 0, 0, 0})
	nw[4][2].MustSetConnectionWeights([]float64{0, 0, 0, 0, 1, 1, 0, 0})
	nw[4][3].MustSetConnectionWeights([]float64{0, 0, 0, 0, 0, 0, 1, 1})

	input := []float64{
		// NOTE(justin): Uncomment ONE of these lines to test the different scenarios
		1, 1, 1, 1, // Solid
		//-1, -1, -1, -1, // Solid (alt)
		//1, 1, -1, -1, // Horizontal
		//-1, -1, 1, 1, // Horizontal (alt)
		//1, -1, -1, 1, // Vertical
		//-1, 1, 1, -1, // Vertical (alt)
		//1, -1, 1, -1, // Diagonal
		//-1, 1, -1, 1, // Diagonal (alt)
	}
	prediction, value, err := nw.Predict(input)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println(prediction, value)
}
