package util

import "fmt"

const DefaultPadding = 10

type Logger struct {
	Name    string
	Padding int
}

func NewLogger(name string, padding int) (nl *Logger) {
	return &Logger{
		Name:    name,
		Padding: padding,
	}
}

func (l Logger) Log(v interface{}) {
	fmt.Printf(fmt.Sprintf("%%%vv: %%#v\n", l.Padding), l.Name, v)
}
