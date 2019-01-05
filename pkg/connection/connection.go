package connection

import (
	"jnet/pkg/data"
	"math/rand"
)

const (
	IndexFrom = iota
	IndexTo
	IndexWeight
)

type Connection [3]data.V

func New(from int, to int) (c *Connection) {
	return &Connection{
		data.V(from), data.V(to), 0,
	}
}

func RandomWeight() (rw data.V) {
	return data.V(rand.Float64())*(data.Max-data.Min) + data.Min
}

type Map []Connection
