package connection

import "jnet/pkg/util"

type Connection struct {
	*util.Logger
	Weight float64
}

func New() (nc *Connection) {
	return &Connection{
		Logger: util.NewLogger("Connection", util.DefaultPadding),
	}
}

type Map [3]float64

type MapCollection []Map
