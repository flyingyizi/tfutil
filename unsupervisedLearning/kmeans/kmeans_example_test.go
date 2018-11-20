package kmeans

import (
	"fmt"
	"path"
	"reflect"
	"testing"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/csvdata"

	"gonum.org/v1/gonum/mat"
)

func ExampleKmeans() {
	//load training X
	filename := "ex7data2.txt"
	X := mat.NewDense(csvdata.CsvToArray(path.Join("testdata", "X"+filename)))
	m, _ := X.Dims()

	//assign original scatter plot data
	origScatt := &tfutil.ScatterData{}
	xs, ys, zs := make([]float64, m), make([]float64, m), make([]float64, m)
	for i := 0; i < m; i++ {
		xs[i], ys[i] = X.At(i, 0), X.At(i, 1)
	}
	origScatt.Add("orig", xs, ys, zs)

	k := 3
	idx := Kmeans(X, k, nil, 100)

	//assign original scatter plot data
	clusterScatt := &tfutil.ScatterData{}
	for i := 0; i < m; i++ {
		xs[i], ys[i], zs[i] = X.At(i, 0), X.At(i, 1), float64(idx[i])
	}
	clusterScatt.Add("cluster", xs, ys, zs)

	tfutil.SaveScatters(filename, origScatt, clusterScatt)
	fmt.Println("")
	//output:
	//
}

func TestKmeans(t *testing.T) {
	type args struct {
		X             *mat.Dense
		k             int
		initCentroids *mat.Dense
		numEpochs     int
	}
	tests := []struct {
		name    string
		args    args
		wantIdx []int
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotIdx := Kmeans(tt.args.X, tt.args.k, tt.args.initCentroids, tt.args.numEpochs); !reflect.DeepEqual(gotIdx, tt.wantIdx) {
				t.Errorf("Kmeans() = %v, want %v", gotIdx, tt.wantIdx)
			}
		})
	}
}
