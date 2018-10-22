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
	theta := mat.NewDense(1, c, nil)

	// X shape is r x c, thea shape is 1 x c, y shape is r x 1

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
	theta := mat.NewDense(1, c, nil)

	// X shape is r x c, theta shape is 1 x c, y shape is r x 1
	k, _ := gradientDescent(X, y, theta, 0.01, 11 /* alpha float64, inters int */)
	fk := mat.Formatted(k, mat.Prefix("    "), mat.Squeeze())
	//fl := mat.Formatted(l, mat.Prefix("    "), mat.Squeeze())
	fmt.Println(fk)
	//	fmt.Println(l)

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
