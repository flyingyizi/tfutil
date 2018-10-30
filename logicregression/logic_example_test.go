package logicregression_test

import (
	"fmt"

	"github.com/flyingyizi/tfutil"
	. "github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/mat"
)

func ExampleComputeCost() {

	X, Y, _ := CsvToDense("ex2data1.txt", false)

	_, rc := X.Dims()
	theta1 := mat.NewVecDense(rc, nil)

	got := ComputeCost(X, Y, theta1)
	fmt.Printf("%.17f\n", got)
	//output:
	//0.69314718055994529
}

func ExampleGradientDescent_ex2data1() {

	X, Y, orig := CsvToDense("ex2data1.txt", true)

	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	t, cost := GradientDescent(X, Y, theta1, 0.01, 1000)
	tfutil.SaveScatter("ex2data1", orig[0], orig[1], t...)
	tfutil.SaveCostLine("ex2data1-cost", cost)

	//fmt.Println(ComputeCost(X, Y, mat.NewVecDense(len(t), t)))
	fmt.Println(t)
	// Output:
	// 0.14744830407944606
}
