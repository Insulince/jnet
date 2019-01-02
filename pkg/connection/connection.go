package connection

const (
	IndexFrom = iota
	IndexTo
	IndexWeight
)

type Connection [3]float64

type Map []Connection
