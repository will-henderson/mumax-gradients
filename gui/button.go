package gui

import (
	"fmt"
)

func (d *Doc) Button(id, value string) string {
	e := newElem(id, value)
	d.add(e)
	return fmt.Sprintf(`<button id=%v onclick="call('%v')">%v</button>`,
		e.id, e.id, "")
}
