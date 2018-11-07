package logicregression_test

import (
	"fmt"
	"math"
	"math/rand"
	"path"
	"time"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/mat"
)

func ExampleGradientDescent_ex2data1() {
	filename := "ex2data1.txt"

	nativ := csvdata.Unflatten(csvdata.CsvToArray(path.Join("testdata", filename), false))
	orig := mat.NewDense(csvdata.Flatten(nativ))
	or, oc := orig.Dims()
	// assign Y
	var Y mat.VecDense
	Y.CloneVec(orig.ColView(oc - 1))
	oldout := make([]float64, Y.Len())
	copy(oldout, Y.RawVector().Data)

	// assign X with additional x_{0}^2 and x_{1}^2 and
	var addition mat.Dense
	addition.Apply(
		func(i, j int, v float64) float64 {
			return math.Pow(v, 2)
		},
		orig.Slice(0, or, 0, oc-1))
	norm := csvdata.FeatureScalingMatrix(orig.Slice(0, or, 0, oc-1))
	additionnorm := csvdata.FeatureScalingMatrix(&addition)
	ones := mat.NewVecDense(or, csvdata.Ones(or))
	X := csvdata.JoinDese(ones, norm, additionnorm)

	//////////////////////////
	x, y, z := make([]float64, 0), make([]float64, 0), make([]float64, 0)
	for i, _ := range oldout {
		x = append(x, orig.At(i, 0))
		y = append(y, orig.At(i, 1))
		z = append(z, orig.At(i, 2))
	}

	s0 := tfutil.ScaterData{}
	s0.Add("m", x, y, z)
	//--end
	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	newtheta, _ := GradientDescent(LogicHyphothesis, X, &Y, theta1, 0.001, 1000)
	newout := Predict(X, mat.NewVecDense(len(newtheta), newtheta))

	// begin to show newout scatter data
	for i, d := range newout {
		x[i], y[i], z[i] = orig.At(i, 0), orig.At(i, 1), float64(d)
	}

	s1 := tfutil.ScaterData{}
	s1.Add("nm", x, y, z)
	//--end

	tfutil.SaveTwoScatter(filename+"new", &s0, &s1)

	//
	// tfutil.SaveCostLine(filename+"cost", cost)
	// diff := 0
	// for i := 0; i < len(newout); i++ {
	// 	if int(oldout[i]) != newout[i] {
	// 		diff++
	// 	}
	// }

	//fmt.Println("total num of training data:", len(newout))
	//fmt.Println("diff num that between orig and predict after training:", diff)
	fmt.Println("")
	//output:
	//
}

func ExampleSaveScatterToImage_ex3data1() {
	filename := "ex3data1.txt"

	r, c, origx := csvdata.CsvToArray(path.Join("testdata", "X"+filename), false)
	orig := mat.NewDense(r, c, origx)

	generateRangeNum := func(min, max int) int {
		rand.Seed(time.Now().Unix())
		randNum := rand.Intn(max-min) + min
		return randNum
	}

	rnd := generateRangeNum(0, (r - 1))
	data := orig.RawRowView(rnd)

	tfutil.SaveScatterToImage(fmt.Sprintf("%s-image-%d", filename, rnd), 20, 20, data)
	//output:
	//
}

func ExampleMultiClassClassification_ex3data1() {
	filename := "ex3data1.txt"

	X := mat.NewDense(csvdata.CsvToArray(path.Join("testdata", "X"+filename), false))
	// Y := func() *mat.VecDense {
	// 	yr, yc, d := csvdata.CsvToArray(path.Join("testdata", "y"+filename), false)
	// 	if yc != 1 {
	// 		return nil
	// 	}
	// 	return mat.NewVecDense(yr, d)
	// }()
	_, _, Y := csvdata.CsvToArray(path.Join("testdata", "y"+filename), false)

	xr, xc := X.Dims()
	ones, norm := mat.NewVecDense(xr, csvdata.Ones(xr)), csvdata.FeatureScalingMatrix(X)
	X = csvdata.JoinDese(ones, norm)
	xr, xc = X.Dims()

	//////////////////////////
	var all_theta []float64
	for label := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10} {
		theta1 := mat.NewVecDense(xc, nil)

		y_i := make([]float64, len(Y))
		for i, j := range Y {
			if label == int(j) {
				y_i[i] = 1
			}
		}
		Y_i := mat.NewVecDense(len(Y), y_i)

		newtheta, _ := GradientDescent(LogicHyphothesis, X, Y_i, theta1, 0.001, 1000)
		all_theta = append(all_theta, newtheta...)
	}
	//////////////////////////

	//Predict(X, mat.NewVecDense(len(newtheta), newtheta))

	//fmt.Println("total num of training data:", len(newout))
	//fmt.Println("diff num that between orig and predict after training:", diff)
	fmt.Println("")
	//output:
	//
}
