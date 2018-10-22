package batch

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func Test_computeCost(t *testing.T) {
	Xd, Yd := loadData("ex1data1.txt")

	r, c := len(Xd), len(Xd[0])
	data := []float64{}
	for _, j := range Xd {
		data = append(data, j...)
	}
	X := mat.NewDense(r, c, data)
	y := mat.NewDense(r, 1, Yd)
	theta := mat.NewDense(1, r, nil)

	cost := computeCost(X, y, theta)
	fmt.Println(cost)

	// type args struct {
	// 	X     *mat.Dense
	// 	y     *mat.Dense
	// 	theta *mat.Dense
	// }
	// tests := []struct {
	// 	name string
	// 	args args
	// 	want float64
	// }{
	// 	// TODO: Add test cases.
	// }
	// for _, tt := range tests {
	// 	t.Run(tt.name, func(t *testing.T) {
	// 		if got := computeCost(tt.args.X, tt.args.y, tt.args.theta); got != tt.want {
	// 			t.Errorf("computeCost() = %v, want %v", got, tt.want)
	// 		}
	// 	})
	// }
}

func Test_gradientDescent(t *testing.T) {
	Xd, Yd := loadData("ex1data1.txt")

	r, c := len(Xd), len(Xd[0])
	data := []float64{}
	for _, j := range Xd {
		data = append(data, j...)
	}
	X := mat.NewDense(r, c, data)
	y := mat.NewDense(r, 1, Yd)
	theta := mat.NewDense(1, r, nil)

	k, l := gradientDescent(X, y, theta, 0.01, 1000 /* alpha float64, inters int */)
	fmt.Println(k, l)

	// type args struct {
	// 	X      *mat.Dense
	// 	y      *mat.Dense
	// 	theta  *mat.Dense
	// 	alpha  float64
	// 	inters int
	// }
	// tests := []struct {
	// 	name       string
	// 	args       args
	// 	wantOtheta *mat.Dense
	// 	wantCost   []float64
	// }{
	// 	// TODO: Add test cases.
	// }
	// for _, tt := range tests {
	// 	t.Run(tt.name, func(t *testing.T) {
	// 		gotOtheta, gotCost := gradientDescent(tt.args.X, tt.args.y, tt.args.theta, tt.args.alpha, tt.args.inters)
	// 		if !reflect.DeepEqual(gotOtheta, tt.wantOtheta) {
	// 			t.Errorf("gradientDescent() gotOtheta = %v, want %v", gotOtheta, tt.wantOtheta)
	// 		}
	// 		if !reflect.DeepEqual(gotCost, tt.wantCost) {
	// 			t.Errorf("gradientDescent() gotCost = %v, want %v", gotCost, tt.wantCost)
	// 		}
	// 	})
	// }
}
