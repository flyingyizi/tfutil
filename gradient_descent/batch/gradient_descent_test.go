package batch

import (
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
			if got := computeCost(tt.args.X, tt.args.y, tt.args.theta); got != tt.want {
				t.Errorf("computeCost() = %v, want %v", got, tt.want)
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

	//k, _ := gradientDescent(X, y, theta, 0.01, 1000 /* alpha float64, inters int */)
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
		wantOtheta *mat.VecDense
		wantCost   float64
	}{
		// TODO: Add test cases.
		{
			name:       "test1",
			args:       args{X: X1, y: y1, theta: theta1, alpha: 0.01, inters: 1000},
			wantOtheta: mat.NewVecDense(2, []float64{-3.2414021442744225, 1.1272942024281842}),
			wantCost:   4.515955503078913,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotOtheta, gotCost := gradientDescent(tt.args.X, tt.args.y, tt.args.theta, tt.args.alpha, tt.args.inters)
			if !reflect.DeepEqual(gotOtheta, tt.wantOtheta) {
				t.Errorf("gradientDescent() gotOtheta = %v, want %v", gotOtheta, tt.wantOtheta)
			} else if got := computeCost(tt.args.X, tt.args.y, gotOtheta); got != tt.wantCost {
				t.Errorf("computeCost() = %v, want %v", got, tt.wantCost)
			}

			if !reflect.DeepEqual(gotCost, tt.wantCost) {
				//	t.Errorf("gradientDescent() gotCost = %v, want %v", gotCost, tt.wantCost)
			}
		})
	}
}
