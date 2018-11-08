package logicregression_test

import (
	"fmt"
	"path"

	"github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func ExampleMultiClassClassification_ex3data1() {
	filename := "ex3data1.txt"

	X := mat.NewDense(csvdata.CsvToArray(path.Join("testdata", "X"+filename), false))
	_, _, Y := csvdata.CsvToArray(path.Join("testdata", "y"+filename), false)

	xr, xc := X.Dims()
	ones, norm := mat.NewVecDense(xr, csvdata.Ones(xr)), csvdata.FeatureScalingMatrix(X)
	X = csvdata.JoinDese(ones, norm)
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

		newtheta, _ := GradientDescent(LogicHyphothesis, X, Y_i, theta1, 0.001, 100, false)
		allTheta[label] = newtheta
	}

	//test
	for k, v := range allTheta {

		result := LogicHyphothesis(X, mat.NewVecDense(len(v), v))
		index := floats.MaxIdx(result.RawVector().Data)

		//k is the label
		if k != Y[index] {
			fmt.Println("bad")
			break
		}
	}
	//output:
	//
}
