package logicregression_test

import (
	"fmt"
	"image/color"
	"math"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/mat"
)

func ExampleGradientDescent_ex2data1() {
	filename := "ex2data1.txt"
	nativ := csvdata.Unflatten(csvdata.CsvToArray("testdata/"+filename, false))
	orig := mat.NewDense(csvdata.Flatten(nativ))
	or, oc := orig.Dims()
	// assign Y
	var Y mat.VecDense
	Y.CloneVec(orig.ColView(oc - 1))
	oldout := make([]float64, Y.Len())
	copy(oldout, Y.RawVector().Data)

	// assign X
	norm := csvdata.FeatureScalingMatrix(orig.Slice(0, or, 0, oc-1))
	ones := mat.NewVecDense(or, csvdata.Ones(or))
	X := csvdata.JoinDese(ones, norm) //X shape is: 'or by (oc)'

	//////////////////////////
	x0, y0 := make([]float64, 0), make([]float64, 0)
	x1, y1 := make([]float64, 0), make([]float64, 0)
	for i, d := range oldout {
		if d == 0 {
			x0 = append(x0, orig.At(i, 0))
			y0 = append(y0, orig.At(i, 1))
		} else {
			x1 = append(x1, orig.At(i, 0))
			y1 = append(y1, orig.At(i, 1))
		}
	}

	s0 := tfutil.ScaterData{}
	s0.Add("m", x0, y0)
	s0.SetColor("m", color.RGBA{R: 255, B: 128, A: 0})
	s0.Add("p", x1, y1)
	//--end
	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	newtheta, _ := GradientDescent(LogicHyphothesis, X, &Y, theta1, 0.001, 1000)
	newout := Predict(X, mat.NewVecDense(len(newtheta), newtheta))

	// begin to show newout scatter data
	newx0, newy0 := make([]float64, 0), make([]float64, 0)
	newx1, newy1 := make([]float64, 0), make([]float64, 0)
	for i, d := range newout {
		if d == 0 {
			newx0 = append(newx0, orig.At(i, 0))
			newy0 = append(newy0, orig.At(i, 1))
		} else {
			newx1 = append(newx1, orig.At(i, 0))
			newy1 = append(newy1, orig.At(i, 1))
		}
	}

	s1 := tfutil.ScaterData{}
	s1.Add("m", newx0, newy0)
	s1.SetColor("m", color.RGBA{R: 255, B: 0, A: 0})
	s1.Add("p", newx1, newy1)
	//--end

	tfutil.SaveTwoScatter(filename+"new", &s0, &s1)

	diff := 0
	for i := 0; i < len(newout); i++ {
		if int(oldout[i]) != newout[i] {
			diff++
		}
	}

	//fmt.Println("total num of training data:", len(newout))
	//fmt.Println("diff num that between orig and predict after training:", diff)
	fmt.Println("")
	// Output:
	//
}

func ExampleGradientDescent_ex2data2() {
	filename := "ex2data2.txt"
	nativ := csvdata.Unflatten(csvdata.CsvToArray("testdata/"+filename, false))
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
	x0, y0 := make([]float64, 0), make([]float64, 0)
	x1, y1 := make([]float64, 0), make([]float64, 0)
	for i, d := range oldout {
		if d == 0 {
			x0 = append(x0, orig.At(i, 0))
			y0 = append(y0, orig.At(i, 1))
		} else {
			x1 = append(x1, orig.At(i, 0))
			y1 = append(y1, orig.At(i, 1))
		}
	}

	s0 := tfutil.ScaterData{}
	s0.Add("m", x0, y0)
	s0.SetColor("m", color.RGBA{R: 255, B: 128, A: 0})
	s0.Add("p", x1, y1)
	//--end
	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	newtheta, cost := GradientDescent(LogicHyphothesis, X, &Y, theta1, 0.001, 1000)
	newout := Predict(X, mat.NewVecDense(len(newtheta), newtheta))

	// begin to show newout scatter data
	newx0, newy0 := make([]float64, 0), make([]float64, 0)
	newx1, newy1 := make([]float64, 0), make([]float64, 0)
	for i, d := range newout {
		if d == 0 {
			newx0 = append(newx0, orig.At(i, 0))
			newy0 = append(newy0, orig.At(i, 1))
		} else {
			newx1 = append(newx1, orig.At(i, 0))
			newy1 = append(newy1, orig.At(i, 1))
		}
	}

	s1 := tfutil.ScaterData{}
	s1.Add("m", newx0, newy0)
	s1.SetColor("m", color.RGBA{R: 255, B: 128, A: 0})
	//s1.SetColor("m", color.RGBA{R: 255, G: 0, B: 0, A: 0})
	s1.Add("p", newx1, newy1)
	//--end

	tfutil.SaveTwoScatter(filename+"new", &s0, &s1)

	//
	tfutil.SaveCostLine(filename+"cost", cost)
	diff := 0
	for i := 0; i < len(newout); i++ {
		if int(oldout[i]) != newout[i] {
			diff++
		}
	}

	//fmt.Println("total num of training data:", len(newout))
	//fmt.Println("diff num that between orig and predict after training:", diff)
	fmt.Println("")
	//output:
	//
}
