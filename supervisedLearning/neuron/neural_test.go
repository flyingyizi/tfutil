package neuron

import (
	"fmt"
	"path"
	"testing"

	"github.com/flyingyizi/tfutil/df"
	"gonum.org/v1/gonum/mat"
)

func TestNeuralNet_ForwardPropa(t *testing.T) {

	//load training X and Y
	X, Y := func() (x *mat.Dense, y []float64) {
		filename := "ex3data1.txt"
		x = mat.NewDense(df.CsvToArray(path.Join("testdata", "X"+filename)))
		_, _, y = df.CsvToArray(path.Join("testdata", "y"+filename))
		return
	}()
	yOneHot := df.EncodeOneHot(10, Y)
	fmt.Printf("X's shape:")
	fmt.Println(X.Dims())
	fmt.Printf("Y's shape:")
	fmt.Println(len(Y))

	nn := NewNetwork(3, 400, 25, 10)
	// layer-1 z: nil	;        layer-1 a:	401 5000 ;
	// layer-2 z:	25 5000  ;   layer-2 a:	26 5000
	// layer-3 z:	10 5000 ;	 layer-3 a:	10 5000
	// layer-2 theta:25 401  ;  layer-3 theta:10 26

	// training numEpochs
	for i := range []int{1, 2, 3, 4} {
		//training use all training sample
		nn.ForwardPropa(X.T())
		grads := nn.BackPropa(X.T(), yOneHot)

		// var y mat.Dense
		// y.Clone(yOneHot.ColView(1))
		// onex := X.RowView(1)
		// nn.ForwardPropa(onex)
		// grads := nn.BackPropa(onex, &y)

		//update each layer's weight with grad
		for i := 1; i < nn.LayerNum; i++ {
			nn.Layers[i].theta.Sub(
				nn.Layers[i].theta,
				grads[i])

			//fk := mat.Formatted(nn.Layers[i].theta, mat.Prefix(""), mat.Squeeze())
			//fmt.Println(fk)
		}
		fmt.Printf("ecpoh:%d \n", i)
		fmt.Println(nn.Cost(X.T(), yOneHot))

	}
	// type args struct {
	// 	x *mat.Dense
	// }
	// tests := []struct {
	// 	name    string
	// 	nn      *NeuralNet
	// 	args    args
	// 	wantErr bool
	// }{
	// 	// TODO: Add test cases.
	// }
	// for _, tt := range tests {
	// 	t.Run(tt.name, func(t *testing.T) {
	// 		if err := tt.nn.ForwardPropa(tt.args.x); (err != nil) != tt.wantErr {
	// 			t.Errorf("NeuralNet.ForwardPropa() error = %v, wantErr %v", err, tt.wantErr)
	// 		}
	// 	})
	// }
}
