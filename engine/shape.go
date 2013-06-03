package engine

import "math"

func init() {
	world.Func("Ellipsoid", Ellipsoid)
	world.Func("Cylinder", Cylinder)
	world.Func("Rect", Rect)
	world.Func("Transl", Transl)
	world.Func("Union", Union)
	world.Func("Intersect", Intersect)
	world.Func("Inverse", Inverse)
}

// geometrical shape for setting sample geometry
type Shape func(x, y, z float64) bool

// Primitives:

// Ellipsoid with given diameters
func Ellipsoid(diamx, diamy, diamz float64) Shape {
	return func(x, y, z float64) bool {
		return sqr64(x/diamx)+sqr64(y/diamy)+sqr64(z/diamz) <= 0.25
	}
}

// Elliptic cylinder along z, with given diameters along x and y.
func Cylinder(diamx, diamy float64) Shape {
	return Ellipsoid(diamx, diamy, inf)
}

// Rectangular slab with given sides.
func Rect(sidex, sidey, sidez float64) Shape {
	return func(x, y, z float64) bool {
		rx, ry, rz := sidex/2, sidey/2, sidez/2
		return x < rx && x > -rx && y < ry && y > -ry && z < rz && z > -rz
	}
}

// Part of space with x > 0.
func HalfSpace() Shape {
	return func(x, y, z float64) bool {
		return x > 0
	}
}

// Transforms:

// Transl returns a translated copy of the shape.
func Transl(s Shape, dx, dy, dz float64) Shape {
	return func(x, y, z float64) bool {
		return s(x-dx, y-dy, z-dz)
	}
}

// CSG:

// Union of shapes a and b.
func Union(a, b Shape) Shape {
	return func(x, y, z float64) bool {
		return a(x, y, z) || b(x, y, z)
	}
}

// Intersection of shapes a and b.
func Intersect(a, b Shape) Shape {
	return func(x, y, z float64) bool {
		return a(x, y, z) && b(x, y, z)
	}
}

// Inverse (outside) of shape.
func Inverse(s Shape) Shape {
	return func(x, y, z float64) bool {
		return !s(x, y, z)
	}
}

// utils

func sqr64(x float64) float64 { return x * x }

var inf = math.Inf(1)