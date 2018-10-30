package logicregression_test

import (
	"testing"

	. "github.com/flyingyizi/tfutil/csvdata"
	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/mat"
)

func TestComputeCost(t *testing.T) {

	X, Y, _ := CsvToDense("ex2data1.txt", false)

	_, rc := X.Dims()
	theta1 := mat.NewVecDense(rc, nil)

	type args struct {
		X     *mat.Dense
		y     *mat.VecDense
		theta mat.Vector
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		// TODO: Add test cases.
		{name: "11",
			args: args{X: X, y: Y, theta: theta1},
			want: 0.69314718055994529},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ComputeCost(tt.args.X, tt.args.y, tt.args.theta); got != tt.want {
				t.Errorf("ComputeCost() = %v, want %v", got, tt.want)
			}
		})
	}
}
