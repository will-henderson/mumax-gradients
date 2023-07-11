module github.com/will-henderson/mumax-gradients

go 1.19

require github.com/mumax/3 v3.9.3+incompatible

//replace github.com/mumax/3 => github.com/will-henderson/mumax-gradients v3.9.3-0.20230526143123-dc7c883db5ab+incompatible
replace github.com/mumax/3 => ./mumax-gradients
