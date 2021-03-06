package linearregression_test

import (
	"reflect"
	"testing"

	"github.com/flyingyizi/tfutil/df"
	. "github.com/flyingyizi/tfutil/linearregression"
	"gonum.org/v1/gonum/mat"
)

//	"gonum.org/v1/gonum/stat"

func Test_gradientDescent(t *testing.T) {
	filename := "ex2data1.txt"
	orig := mat.NewDense(df.CsvToArray("testdata/" + filename))
	or, oc := orig.Dims()
	// assign Y
	var Y mat.VecDense
	Y.CloneVec(orig.ColView(oc - 1))
	// assign Y
	ones := mat.NewVecDense(or, df.Ones(or))
	X := df.HorizJoinDense(ones, orig.Slice(0, or, 0, oc-1)) //X shape is: 'or by (oc)'

	_, rc := X.Dims()
	theta1 := mat.NewVecDense(rc, nil)

	// 	//k, _ := BGD(X, y, theta, 0.01, 1000 /* alpha float64, inters int */)
	// 	//fk := mat.Formatted(k, mat.Prefix("    "), mat.Squeeze())
	// 	//fmt.Println(fk)

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
			args:       args{X: X, y: &Y, theta: theta1, alpha: 0.01, inters: 1000},
			wantOtheta: []float64{-1.0199878766662607e-16, 0.8785036522230538, -0.04691665703805384},
			//wantCost:   0.13070336960771892,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotOtheta, gotCost := BGD(tt.args.X, tt.args.y, tt.args.theta, tt.args.alpha, tt.args.inters)
			if !reflect.DeepEqual(gotOtheta, tt.wantOtheta) {
				t.Errorf("BGD() gotOtheta = %v, want %v", gotOtheta, tt.wantOtheta)
			}

			if !reflect.DeepEqual(gotCost, tt.wantCost) {
				//t.Errorf("BGD() gotCost = %v, want %v", gotCost, tt.wantCost)
			}
		})
	}
}
