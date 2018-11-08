package neuron_test

import (
	"fmt"
	"math"
	"path"

	"gonum.org/v1/gonum/floats"

	"github.com/flyingyizi/tfutil/csvdata"

	"gonum.org/v1/gonum/mat"
)

func ExampleFeedForwardPrediction_ex3weights() {

	//load theta
	Theta1, Theta2 := func() (theta1, theta2 *mat.Dense) {
		filename := "ex3weights.txt"
		theta1 = mat.NewDense(csvdata.CsvToArray(path.Join("testdata", "Theta1"+filename), false))
		theta2 = mat.NewDense(csvdata.CsvToArray(path.Join("testdata", "Theta2"+filename), false))
		return
	}()
	fmt.Printf("Theta1's shape:")
	fmt.Println(Theta1.Dims())
	fmt.Printf("Theta2's shape:")
	fmt.Println(Theta2.Dims())

	//load training X and Y
	X, Y := func() (x *mat.Dense, y []float64) {
		filename := "ex3data1.txt"
		x = mat.NewDense(csvdata.CsvToArray(path.Join("testdata", "X"+filename), false))
		_, _, y = csvdata.CsvToArray(path.Join("testdata", "y"+filename), false)
		return
	}()
	fmt.Printf("X's shape:")
	fmt.Println(X.Dims())
	fmt.Printf("Y's shape:")
	fmt.Println(len(Y))

	//layer-1 a1:= [ones, X], it is neuron input
	a1 := func() *mat.Dense {
		xr, _ := X.Dims()
		z := csvdata.JoinDese(mat.NewVecDense(xr, csvdata.Ones(xr)), X)
		return z
	}()
	fmt.Printf("a1's shape:")
	fmt.Println(a1.Dims())

	//layer-2 a2:= sigmod(a1 * Theta1.T)
	a2 := func() *mat.Dense {
		var z mat.Dense
		z.Mul(a1, Theta1.T())

		z.Apply(
			func(i, j int, v float64) float64 {
				return 1 / 1 / (1 + math.Exp(-v))
			}, &z)

		return &z
	}()
	fmt.Printf("a2's shape:")
	fmt.Println(a2.Dims())

	//layer-3 a3:= sigmod( [ones,a2] * Theta2.T)
	a3 := func() *mat.Dense {
		a2r, _ := a2.Dims()
		ones, norm := mat.NewVecDense(a2r, csvdata.Ones(a2r)), csvdata.FeatureScalingMatrix(a2)
		ones_a2 := csvdata.JoinDese(ones, norm)

		var z mat.Dense
		z.Mul(ones_a2, Theta2.T())

		z.Apply(
			func(i, j int, v float64) float64 {
				return 1 / 1 / (1 + math.Exp(-v))
			}, &z)

		return &z
	}()
	fmt.Printf("a3's shape:")
	fmt.Println(a3.Dims())

	//a3 store the  probability result, for 10 labels
	// we will calc the max probalility for each label
	a3r, _ := a3.Dims()
	yPred := make([]float64, a3r)
	for i := 0; i < a3r; i++ {
		yPred[i] = float64(floats.MaxIdx(a3.RawRowView(i)) + 1) //0 base index, +1 for label convention
	}

	//compare the predict label and training label, calc the hit rate
	percent := func() float64 {
		floats.Sub(yPred, Y)
		sum := 0.0
		for _, j := range yPred {
			if j == 0 {
				sum++
			}
		}
		return sum * 100 / float64(len(yPred))
	}()

	fmt.Printf("accuracy: %.2f %s \n", percent, "%")

	//output:
	// Theta1's shape:25 401
	// Theta2's shape:10 26
	// X's shape:5000 400
	// Y's shape:5000
	// a1's shape:5000 401
	// a2's shape:5000 25
	// a3's shape:5000 10
	// accuracy: 97.10 %
}
