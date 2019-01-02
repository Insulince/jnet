package connection

import (
	"jnet/pkg/util"
	"math/rand"
)

const (
	IndexFrom = iota
	IndexTo
	IndexWeight
)

type Connection [3]float64

func New(from int, to int) (c *Connection) {
	return &Connection{
		float64(from), float64(to), RandomWeight(),
	}
}

func RandomWeight() (rw float64) {
	return rand.Float64()*(util.Max-util.Min) + util.Min
}

type Map []Connection
