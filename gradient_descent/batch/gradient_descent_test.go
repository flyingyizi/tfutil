package batch

import (
	//	"gonum.org/v1/gonum/stat"
	"reflect"
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
	X1 := mat.NewDense(r, c, data)
	y1 := mat.NewVecDense(r, Yd)
	theta1 := mat.NewVecDense(c, nil)

	type args struct {
		X     *mat.Dense
		y     *mat.VecDense
		theta *mat.VecDense
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		// TODO: Add test cases.
		{
			name: "test1",
			args: args{X: X1, y: y1, theta: theta1},
			want: 32.072733877455676,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ComputeCost(tt.args.X, tt.args.y, tt.args.theta); got != tt.want {
				t.Errorf("ComputeCost() = %v, want %v", got, tt.want)
			}
		})
	}
}
func Test_gradientDescent(t *testing.T) {
	Xd, Yd := loadData("ex1data1.txt")

	data := []float64{}
	for _, j := range Xd {
		data = append(data, 1)
		data = append(data, j...)
	}
	r, c := len(Xd), len(Xd[0])+1
	X1 := mat.NewDense(r, c, data)
	y1 := mat.NewVecDense(r, Yd)
	theta1 := mat.NewVecDense(c, nil)

	//k, _ := GradientDescent(X, y, theta, 0.01, 1000 /* alpha float64, inters int */)
	//fk := mat.Formatted(k, mat.Prefix("    "), mat.Squeeze())
	//fmt.Println(fk)

	type args struct {
		X      *mat.Dense
		y      *mat.VecDense
		theta  *mat.VecDense
		alpha  float64
		inters int
	}
	tests := []struct {
		name       string
		args       args
		wantOtheta []float64
		wantCost   float64
	}{
		// TODO: Add test cases.
		{
			name:       "test1",
			args:       args{X: X1, y: y1, theta: theta1, alpha: 0.01, inters: 1000},
			wantOtheta: []float64{-3.2414021442744225, 1.1272942024281842},
			wantCost:   4.515955503078913,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotOtheta, gotCost := GradientDescent(tt.args.X, tt.args.y, tt.args.theta, tt.args.alpha, tt.args.inters)
			if !reflect.DeepEqual(gotOtheta, tt.wantOtheta) {
				t.Errorf("GradientDescent() gotOtheta = %v, want %v", gotOtheta, tt.wantOtheta)
			}

			if !reflect.DeepEqual(gotCost, tt.wantCost) {
				//	t.Errorf("GradientDescent() gotCost = %v, want %v", gotCost, tt.wantCost)
			}
		})
	}
}
