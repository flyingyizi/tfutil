package logicregression_test

import (
	"fmt"
	"image/color"

	"github.com/flyingyizi/tfutil"
	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/mat"
)

func ExampleComputeCost() {

	X, Y, _ := CsvToDense("ex2data1.txt", false)

	_, rc := X.Dims()
	theta1 := mat.NewVecDense(rc, nil)

	got := ComputeCost(LogicHyphothesis, X, Y, theta1)
	fmt.Printf("%.17f\n", got)
	//output:
	//0.69314718055994529
}

func ExampleGradientDescent_ex2data1() {

	X, Y, orig := CsvToDense("ex2data1.txt", true)
	// begin to show orig scatter data
	zeros := 0
	for _, j := range orig[2] {
		if j == 0 {
			zeros++
		}
	}

	x0, y0 := make([]float64, zeros), make([]float64, zeros)
	x1, y1 := make([]float64, len(orig[2])-zeros), make([]float64, len(orig[2])-zeros)

	first, second := 0, 0
	for i := 0; i < len(orig[2]); i++ {
		if orig[2][i] == 0 {
			x0[first], y0[first] = orig[0][i], orig[1][i]
			first++
		} else {
			x1[second], y1[second] = orig[0][i], orig[1][i]
			second++
		}
	}
	s0 := tfutil.ScaterData{}
	s0.Add("m", x0, y0)
	s0.SetColor("m", color.RGBA{R: 255, B: 128, A: 0})
	s0.Add("p", x1, y1)
	//--end
	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	t, _ := GradientDescent(LogicHyphothesis, X, Y, theta1, 0.001, 1000)

	newout := Predict(X, mat.NewVecDense(len(t), t))

	// begin to show newout scatter data
	zeros = 0
	for _, j := range newout {
		if j == 0 {
			zeros++
		}
	}

	x0, y0 = make([]float64, zeros), make([]float64, zeros)
	x1, y1 = make([]float64, len(newout)-zeros), make([]float64, len(newout)-zeros)

	first, second = 0, 0
	for i := 0; i < len(newout); i++ {
		if newout[i] == 0 {
			x0[first], y0[first] = orig[0][i], orig[1][i]
			first++
		} else {
			x1[second], y1[second] = orig[0][i], orig[1][i]
			second++
		}
	}
	s1 := tfutil.ScaterData{}
	s1.Add("m", x0, y0)
	s1.SetColor("m", color.RGBA{R: 255, B: 0, A: 0})
	s1.Add("p", x1, y1)
	//--end

	tfutil.SaveTwoScatter("ex2data1new", &s0, &s1)

	diff := 0
	for i := 0; i < len(newout); i++ {
		if int(orig[2][i]) != newout[i] {
			diff++
		}
	}

	//fmt.Println(ComputeCost(X, Y, mat.NewVecDense(len(t), t)))
	//fmt.Println(newout)
	//fmt.Println(orig[2])
	fmt.Println("total num of training data:", len(newout))
	fmt.Println("diff num that between orig and predict after training:", diff)
	// Output:
	//
}

func ExampleGradientDescent_ex2data2() {

	X, Y, orig := CsvToDense("ex2data2.txt", true)
	// begin to show orig scatter data
	zeros := 0
	for _, j := range orig[2] {
		if j == 0 {
			zeros++
		}
	}

	x0, y0 := make([]float64, zeros), make([]float64, zeros)
	x1, y1 := make([]float64, len(orig[2])-zeros), make([]float64, len(orig[2])-zeros)

	first, second := 0, 0
	for i := 0; i < len(orig[2]); i++ {
		if orig[2][i] == 0 {
			x0[first], y0[first] = orig[0][i], orig[1][i]
			first++
		} else {
			x1[second], y1[second] = orig[0][i], orig[1][i]
			second++
		}
	}
	s0 := tfutil.ScaterData{}
	s0.Add("m", x0, y0)
	s0.SetColor("m", color.RGBA{R: 255, B: 128, A: 0})
	s0.Add("p", x1, y1)
	//--end
	_, xc := X.Dims()
	theta1 := mat.NewVecDense(xc, nil)

	t, _ := GradientDescent(LogicHyphothesis, X, Y, theta1, 0.001, 1000)

	newout := Predict(X, mat.NewVecDense(len(t), t))

	// begin to show newout scatter data
	zeros = 0
	for _, j := range newout {
		if j == 0 {
			zeros++
		}
	}

	x0, y0 = make([]float64, zeros), make([]float64, zeros)
	x1, y1 = make([]float64, len(newout)-zeros), make([]float64, len(newout)-zeros)

	first, second = 0, 0
	for i := 0; i < len(newout); i++ {
		if newout[i] == 0 {
			x0[first], y0[first] = orig[0][i], orig[1][i]
			first++
		} else {
			x1[second], y1[second] = orig[0][i], orig[1][i]
			second++
		}
	}
	s1 := tfutil.ScaterData{}
	s1.Add("m", x0, y0)
	s1.SetColor("m", color.RGBA{R: 255, B: 0, A: 0})
	s1.Add("p", x1, y1)
	//--end

	tfutil.SaveTwoScatter("ex2data2new", &s0, &s1)

	diff := 0
	for i := 0; i < len(newout); i++ {
		if int(orig[2][i]) != newout[i] {
			diff++
		}
	}

	//fmt.Println(ComputeCost(X, Y, mat.NewVecDense(len(t), t)))
	//fmt.Println(newout)
	//fmt.Println(orig[2])
	fmt.Println("total num of training data:", len(newout))
	fmt.Println("diff num that between orig and predict after training:", diff)
	// Output:
	//
}
