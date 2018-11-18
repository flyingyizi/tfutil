package logicregression_test

import (
	"fmt"
	"path"

	"github.com/flyingyizi/tfutil"

	"github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func ExampleComputeCost() {

	filename := "ex2data1.txt"
	//load traning data
	X, Y := func() (x *mat.Dense, y *mat.VecDense) {
		orig := mat.NewDense(csvdata.CsvToArray(path.Join("testdata", filename)))
		or, oc := orig.Dims()
		// assign Y
		var Y mat.VecDense
		Y.CloneVec(orig.ColView(oc - 1))
		// assign X
		ones := mat.NewVecDense(or, csvdata.Ones(or))
		X := csvdata.HorizJoinDense(ones, orig.Slice(0, or, 0, oc-1)) //X shape is: 'or by (oc)'
		return X, &Y
	}()

	_, rc := X.Dims()
	theta1 := mat.NewVecDense(rc, nil)

	_, cost := BGD(LogicHyphothesis, X, Y, theta1, 0.001, 100, true)

	tfutil.SaveLine(filename+"cost", cost)
	//output:
	//
}

func ExampleMultiClassClassification_ex3data1() {
	filename := "ex3data1.txt"

	X := mat.NewDense(csvdata.CsvToArray(path.Join("testdata", "X"+filename)))
	_, _, Y := csvdata.CsvToArray(path.Join("testdata", "y"+filename))

	xr, xc := X.Dims()
	ones, norm := mat.NewVecDense(xr, csvdata.Ones(xr)), csvdata.FeatureScalingMatrix(X)
	X = csvdata.HorizJoinDense(ones, norm)
	xr, xc = X.Dims()

	allLabels := map[float64]string{1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
		6: "6", 7: "7", 8: "8", 9: "9", 10: "10"}

	allTheta := make(map[float64][]float64)
	for label, _ := range allLabels {
		theta1 := mat.NewVecDense(xc, nil)

		y_i := make([]float64, len(Y))
		for i, j := range Y {
			if label == j {
				y_i[i] = 1
			}
		}
		Y_i := mat.NewVecDense(len(Y), y_i)

		newtheta, _ := BGD(LogicHyphothesis, X, Y_i, theta1, 0.001, 100, false)
		allTheta[label] = newtheta
	}

	//test
	for k, v := range allTheta {

		result := LogicHyphothesis(X, mat.NewVecDense(len(v), v))

		index := floats.MaxIdx(result.RawMatrix().Data)

		//k is the label
		if k != Y[index] {
			fmt.Println("bad")
			break
		}
	}
	//output:
	//
}
