package jnet

type Connection struct {
	*Logger
	Weight float64
}

func NewConnection() (nc *Connection) {
	return &Connection{
		Logger: NewLogger("Connection", DefaultPadding),
	}
}
