package linearregression_test

import (
	"fmt"

	"github.com/flyingyizi/tfutil"
	. "github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/linearregression"
	"gonum.org/v1/gonum/mat"
)

func ExampleGradientDescent_ex1data1() {

	X, Y, _ := CsvToDense("ex1data1.txt", true)

	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	t, cost := GradientDescent(X, Y, theta1, 0.01, 1000)
	//tfutil.SaveScatter("ex1data1", orig[0], orig[1], t...)
	tfutil.SaveCostLine("ex1data1-cost", cost)

	fmt.Println(ComputeCost(X, Y, mat.NewVecDense(len(t), t)))
	// Output:
	// 0.14744830407944606
}

func ExampleGradientDescent_ex1data2() {

	X, Y, _ := CsvToDense("ex1data2.txt", true)

	_, rc := X.Dims()
	theta1 := mat.NewVecDense(rc, nil)

	t, cost := GradientDescent(X, Y, theta1, 0.01, 1000)
	//SaveScatter("ex1data2", orig[0], orig[1], t...)
	tfutil.SaveCostLine("ex1data2-cost", cost)

	theta1 = mat.NewVecDense(rc, t)
	tfutil.SaveResidualPlot("ex1data2-residual", X, Y, theta1)

	fmt.Println(ComputeCost(X, Y, mat.NewVecDense(len(t), t)))
	// Output:
	// 0.13070336960771892
}
