package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	t := func(x float64) (y float64) {
		return 2*math.Pow(x, 4) + 17*math.Pow(x, 3) - 13*math.Pow(x, 2) + 7*x - 53
	}

	dt := func(x float64) (y float64) {
		return 8*math.Pow(x, 3) + 51*math.Pow(x, 2) - 26*x + 7
	}

	const Amplitude = 10
	const Iterations = 1000
	const LearningRate = 0.001
	const CutOff = 1.0e-12

	x := rand.Float64() * Amplitude

	c := 0
	for i := 0; i < Iterations; i++ {
		tv, dtv := t(x), dt(x)

		if dtv > 0.0 {
			x -= dtv * LearningRate
		} else if dtv < 0.0 {
			x += dtv * LearningRate
		}

		fmt.Printf("%22v - %22v - %22v\n", x, tv, dtv)

		c++

		if math.Abs(dtv) < CutOff {
			fmt.Println("Sufficiently minimized, breaking...")
			break
		}
	}

	fmt.Printf("Minimum weight value is %v\nIterations: %v", x, c)

	os.Exit(0)
}
