package linearregression_test

import (
	"fmt"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/df"
	. "github.com/flyingyizi/tfutil/linearregression"
	"gonum.org/v1/gonum/mat"
)

func ExampleBGD_ex1data1() {
	filename := "ex1data1.txt"
	orig := mat.NewDense(df.CsvToArray("testdata/" + filename))
	or, oc := orig.Dims()
	// assign Y
	var Y mat.VecDense
	Y.CloneVec(orig.ColView(oc - 1))
	// assign X
	ones := mat.NewVecDense(or, df.Ones(or))
	X := df.HorizJoinDense(ones, orig.Slice(0, or, 0, oc-1)) //X shape is: 'or by (oc)'

	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	t, cost := BGD(X, &Y, theta1, 0.01, 1000)
	tfutil.SaveLine(filename+"cost", cost)

	fmt.Println(ComputeCost(X, &Y, mat.NewVecDense(len(t), t)))
	// Output:
	// 0.14744830407944606
}
