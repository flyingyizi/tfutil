package iforest

import (
	"reflect"
	"testing"

	"github.com/flyingyizi/tfutil/df"
	"gonum.org/v1/gonum/mat"
)

func Test_splitSample(t *testing.T) {
	type args struct {
		X         [][]float64
		split     float64
		attribute int
		ind       []int
	}
	tests := []struct {
		args  args
		want  []int
		want1 []int
	}{
		{args: args{
			X:     [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}},
			split: 2, attribute: 0, ind: []int{0, 1, 2, 3}}, want: []int{0, 1}, want1: []int{2, 3}},
		{args: args{
			X:     [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}},
			split: 3, attribute: 1, ind: []int{0, 1, 2, 3}}, want: []int{0, 1, 2}, want1: []int{3}},
		{args: args{
			X:     [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}},
			split: 6, attribute: 1, ind: []int{0, 1, 2, 3}}, want: []int{0, 1, 2, 3}, want1: []int{}},
		{args: args{
			X:     [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}},
			split: 2, attribute: 0, ind: []int{1, 2}}, want: []int{1}, want1: []int{2}},
	}
	for _, tt := range tests {
		a := mat.NewDense(df.Flatten(tt.args.X))

		got, got1 := splitSample(a.T(), tt.args.ind, tt.args.split, tt.args.attribute)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("splitSample() got = %v, want %v", got, tt.want)
		}
		if !reflect.DeepEqual(got1, tt.want1) {
			t.Errorf("splitSample() got1 = %v, want %v", got1, tt.want1)
		}

	}
}

func Test_radomChooseSplit(t *testing.T) {
	type args struct {
		X   [][]float64
		ind []int
		att int
	}
	tests := []struct {
		args  args
		want1 float64
		want2 float64
	}{
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, att: 0, ind: []int{0, 1, 2, 3}}, want1: 1, want2: 3},
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, att: 1, ind: []int{0, 1, 2, 3}}, want1: 2, want2: 5},
	}
	for _, tt := range tests {
		a := mat.NewDense(df.Flatten(tt.args.X))

		if got := radomChooseSplit(a.T(), tt.args.ind, tt.args.att); got < tt.want1 || got > tt.want2 {
			t.Errorf("radomChooseSplit() = %v, want between %v : %v", got, tt.want1, tt.want2)
		}

	}
}

func TestTree_BuildTree(t *testing.T) {
	type args struct {
		X   [][]float64
		ids []int
	}
	tests := []struct {
		t       *Tree
		args    args
		wantErr bool
	}{
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, ids: []int{0, 1, 2, 3}}, t: &Tree{}, wantErr: false},
	}
	for _, tt := range tests {
		a := mat.NewDense(df.Flatten(tt.args.X))
		if err := tt.t.BuildTree(a.T(), tt.args.ids, 2 /*MaxDepth = */); (err != nil) != tt.wantErr && tt.t.Root != nil {
			t.Errorf("Tree.BuildTree() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}
