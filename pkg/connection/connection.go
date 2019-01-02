package connection

import "jnet/pkg/util"

type Connection struct {
	*util.Logger
	Weight float64
}

func NewConnection() (nc *Connection) {
	return &Connection{
		Logger: util.NewLogger("Connection", util.DefaultPadding),
	}
}
