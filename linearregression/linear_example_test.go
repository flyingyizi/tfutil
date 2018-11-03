package linearregression_test

import (
	"fmt"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/linearregression"
	"gonum.org/v1/gonum/mat"
)

func ExampleGradientDescent_ex1data1() {
	filename := "ex1data1.txt"
	orig := mat.NewDense(csvdata.CsvToArray("testdata/"+filename, true))
	or, oc := orig.Dims()
	// assign Y
	var Y mat.VecDense
	Y.CloneVec(orig.ColView(oc - 1))
	// assign Y
	ones := mat.NewVecDense(or, csvdata.Ones(or))
	X := csvdata.JoinDese(ones, orig.Slice(0, or, 0, oc-1)) //X shape is: 'or by (oc)'

	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	t, cost := GradientDescent(X, &Y, theta1, 0.01, 1000)
	//tfutil.SaveScatter("ex1data1", orig[0], orig[1], t...)
	tfutil.SaveCostLine(filename+"cost", cost)

	fmt.Println(ComputeCost(X, &Y, mat.NewVecDense(len(t), t)))
	// Output:
	// 0.14744830407944606
}

func ExampleGradientDescent_ex1data2() {
	filename := "ex1data2.txt"
	orig := mat.NewDense(csvdata.CsvToArray("testdata/"+filename, true))
	or, oc := orig.Dims()
	// assign Y
	var Y mat.VecDense
	Y.CloneVec(orig.ColView(oc - 1))
	// assign Y
	ones := mat.NewVecDense(or, csvdata.Ones(or))
	X := csvdata.JoinDese(ones, orig.Slice(0, or, 0, oc-1)) //X shape is: 'or by (oc)'

	_, rc := X.Dims()
	theta1 := mat.NewVecDense(rc, nil)

	t, cost := GradientDescent(X, &Y, theta1, 0.01, 1000)
	//SaveScatter("ex1data2", orig[0], orig[1], t...)
	tfutil.SaveCostLine(filename+"cost", cost)

	theta1 = mat.NewVecDense(rc, t)
	tfutil.SaveResidualPlot(filename+"-residual", X, &Y, theta1)

	fmt.Println(ComputeCost(X, &Y, mat.NewVecDense(len(t), t)))
	// Output:
	// 0.13070336960771892
}
