package main

import (
	"fmt"
	"github.com/Insulince/jnet"
)

func main() {
	nm := []int{3, 3, 3}
	il := []string{"a", "b", "c"}
	ol := []string{"0", "1", "2"}
	nw1, err := jnet.NewNetwork(nm, il, ol)
	if err != nil {
		panic(err)
	}
	s1 := nw1.Serialize()
	fmt.Println(s1)
	fmt.Println()
	nw2, err := jnet.Deserialize(s1)
	if err != nil {
		panic(err)
	}
	s2 := nw2.Serialize()
	fmt.Println(s2)
	fmt.Println()
	fmt.Println(s1 == s2)
}
