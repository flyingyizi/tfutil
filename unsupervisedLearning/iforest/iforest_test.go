package iforest

import (
	"testing"

	"github.com/flyingyizi/tfutil/df"
	"gonum.org/v1/gonum/mat"
)

func TestForest_Train(t *testing.T) {
	tests := []struct {
		f *Forest
		X [][]float64
	}{
		{f: NewForest(AnomalyRatio(0.1), NbTrees(1), SubsamplingSize(2)),
			X: [][]float64{
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1}}},
	}
	for _, tt := range tests {

		a := mat.NewDense(df.Flatten(tt.X))
		tt.f.Train(a.T())
	}
}

func TestForest_Test(t *testing.T) {
	tests := []struct {
		X          [][]float64
		f          *Forest
		trainFirst bool
		wantErr    bool
	}{
		{f: NewForest(AnomalyRatio(0.4), NbTrees(1), SubsamplingSize(4)),
			X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: false, wantErr: true},
		{f: NewForest(AnomalyRatio(0.4), NbTrees(1), SubsamplingSize(4)),
			X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: true, wantErr: false},
	}
	for _, tt := range tests {

		a := mat.NewDense(df.Flatten(tt.X))
		if tt.trainFirst {
			tt.f.Train(a)
		}

		if err := tt.f.Test(a); (err != nil) != tt.wantErr {
			t.Errorf("Forest.Test() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

func TestForest_Predict(t *testing.T) {

	tests := []struct {
		f          *Forest
		trainFirst bool
		testFirst  bool
		X          [][]float64
		wantErr    bool
	}{
		{f: NewForest(AnomalyRatio(0.4), NbTrees(1), SubsamplingSize(4)),
			X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: false, testFirst: false, wantErr: true},
		{f: NewForest(AnomalyRatio(0.4), NbTrees(1), SubsamplingSize(4)),
			X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: true, testFirst: false, wantErr: true},
		{f: NewForest(AnomalyRatio(0.4), NbTrees(1), SubsamplingSize(4)),
			X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: true, testFirst: true, wantErr: false},
	}
	for _, tt := range tests {
		a := mat.NewDense(df.Flatten(tt.X))

		if tt.trainFirst {
			tt.f.Train(a)
		}
		if tt.testFirst {
			tt.f.Test(a)
		}
		_, _, err := tt.f.Predict(a)
		if (err != nil) != tt.wantErr {
			t.Errorf("Forest.Predict() error = %v, wantErr %v", err, tt.wantErr)
			return
		}
	}
}

func TestForest_Save(t *testing.T) {

	tests := []struct {
		name    string
		f       *Forest
		path    string
		wantErr bool
	}{
		{f: NewForest(AnomalyRatio(0.1), NbTrees(1), SubsamplingSize(2)), path: "", wantErr: true},
		{f: NewForest(AnomalyRatio(0.1), NbTrees(1), SubsamplingSize(2)), path: "aaa", wantErr: false},
	}
	for _, tt := range tests {

		if err := tt.f.Save(tt.path); (err != nil) != tt.wantErr {
			t.Errorf("Forest.Save() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

func TestForest_Load(t *testing.T) {

	tests := []struct {
		name    string
		f       *Forest
		path    string
		wantErr bool
	}{
		{f: NewForest(AnomalyRatio(0.1), NbTrees(1), SubsamplingSize(2)), path: "", wantErr: true},
		{f: NewForest(AnomalyRatio(0.1), NbTrees(1), SubsamplingSize(2)), path: "xD", wantErr: true},
		{f: NewForest(AnomalyRatio(0.1), NbTrees(1), SubsamplingSize(2)), path: "aaa", wantErr: false},
	}
	for _, tt := range tests {

		if err := tt.f.Load(tt.path); (err != nil) != tt.wantErr {
			t.Errorf("Forest.Load() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}
