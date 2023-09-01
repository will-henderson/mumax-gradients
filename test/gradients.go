//go:build ignore
// +build ignore

package main

import (
	en "github.com/mumax/3/engine"
)

func main() {

	defer en.InitAndClose()()

}
