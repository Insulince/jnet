package neuron

import (
	"github.com/TheDemx27/calculus"
	"math"
)

type NeuralTransformer func(x float64) (y float64)

func sigmoid(x float64) (y float64) {
	return float64(2/(1+math.Pow(math.E, float64(-x))) - 1)
}

func dSigmoid(x float64) (y float64) {
	return calculus.Diff(sigmoid, x)
}

func rectifiedLinearUnit(x float64) (y float64) {
	if x > 0 {
		return x
	}
	return 0
}

func dRectifiedLinearUnit(x float64) (y float64) {
	return calculus.Diff(rectifiedLinearUnit, x)
}

func passThrough(x float64) (y float64) {
	return x
}

func dPassThrough(x float64) (y float64) {
	return calculus.Diff(passThrough, x)
}

func twoTimes(x float64) (y float64) {
	return 2 * x
}

func dTwoTimes(x float64) (y float64) {
	return calculus.Diff(twoTimes, x)
}

func exp(x float64) (y float64) {
	return float64(math.Pow(math.E, float64(x)))
}

func dExp(x float64) (y float64) {
	return calculus.Diff(exp, x)
}

func onePlus(x float64) (y float64) {
	return 1 + x
}

func dOnePlus(x float64) (y float64) {
	return calculus.Diff(onePlus, x)
}

func inverse(x float64) (y float64) {
	return 1.0 / x
}

func dInverse(x float64) (y float64) {
	return calculus.Diff(inverse, x)
}

func negate(x float64) (y float64) {
	return -1 * x
}

func dNegate(x float64) (y float64) {
	return calculus.Diff(negate, x)
}
