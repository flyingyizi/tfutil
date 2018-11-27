package neuron

import (
	"math"
	"math/rand"
	"time"

	"github.com/flyingyizi/tfutil/df"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

// NeuralNet contains all of the information
// that defines a trained neural network.
type NeuralNet struct {
	LayerNum int
	Layers   []Layer
}

//Layer represent nn's layer
type Layer struct {
	n     int // num of neural
	theta *mat.Dense
	z, a  *mat.Dense
}

// NewNetwork initializes a new neural network. fill the weight marix with rand
//layerNum is num of layer
//sls is list of neural's num in each layser
func NewNetwork(layerNum int, sls ...int) *NeuralNet {

	if len(sls) != layerNum {
		return nil
	}
	nn := NeuralNet{LayerNum: layerNum,
		Layers: make([]Layer, layerNum)}

	for i := 0; i < layerNum; i++ {
		nn.Layers[i].n = sls[i]
	}

	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	// $W^{(l)} \in \mathbb{R}^{S_l \times S_{(l-1)}} $：表示$l − 1$ 层到第$l$ 层的权重矩阵；
	for i := 1; i < layerNum; i++ {
		curr, prev := &(nn.Layers[i]), &(nn.Layers[i-1])
		tr, tc := curr.n, prev.n+1
		curr.theta = mat.NewDense(tr, tc, nil)

		curr.theta.Apply(
			func(i, j int, v float64) float64 {
				return randGen.Float64()
			}, curr.theta)

	}
	return &nn
}

//ForwardPropa feed forward propagate
func (nn *NeuralNet) ForwardPropa(x mat.Matrix) error {
	//             a1       a2      a3
	//z shape:  400x5000,   25x5000,     10x5000
	//a shape: (400+1)x5000, (25+1)x5000, (10+0)x5000
	//w shape:               25x401,        10x26

	_, xc := x.Dims()
	ones := df.Ones(xc)

	for i := 0; i < nn.LayerNum; i++ {
		curr := &(nn.Layers[i])
		//input layer
		if i == 0 {
			curr.z = nil
			curr.a = df.VerticalJoinDense(mat.NewDense(1, len(ones), ones), x)
			continue
		}

		prev := &(nn.Layers[i-1])
		curr.z = func() *mat.Dense {
			var z mat.Dense
			z.Mul(curr.theta, prev.a)
			return &z
		}()
		curr.a = func() *mat.Dense {
			var a mat.Dense
			a.Apply(
				func(i, j int, v float64) float64 {
					return sigmoid(v)
				}, curr.z)
			if i == nn.LayerNum-1 {
				return &a
			}
			pa := df.VerticalJoinDense(mat.NewDense(1, len(ones), ones), &a)
			return pa
		}()

	}
	return nil
}

func vectorToSlice(a mat.Vector) (r []float64) {
	var vd mat.VecDense
	vd.CloneVec(a)

	r = make([]float64, a.Len())
	copy(r, vd.RawVector().Data)
	return
}

//Cost  claculate cost
// before execute cost, the nn ForwardPropa should be executed firstly
// = - ( y .∗ log(hθ) + (1−y) .∗ log(1−hθ) )
func (nn *NeuralNet) Cost(x mat.Matrix, yOnehot *mat.Dense) float64 {
	_, m := x.Dims()

	logh0 := func() *mat.Dense {
		var t mat.Dense
		t.Apply(
			func(i, j int, v float64) float64 {
				return math.Log(v)
			}, nn.Layers[nn.LayerNum-1].a)
		return &t
	}()
	logh1 := func() *mat.Dense {
		var t mat.Dense
		t.Apply(
			func(i, j int, v float64) float64 {
				return math.Log(1 - v)
			}, nn.Layers[nn.LayerNum-1].a)
		return &t
	}()

	first := func() *mat.Dense {
		var t mat.Dense
		t.MulElem(yOnehot, logh0)
		return &t
	}()
	second := func() *mat.Dense {
		var t mat.Dense
		t.Apply(
			func(i, j int, v float64) float64 {
				return 1 - v
			}, yOnehot)
		t.MulElem(&t, logh1)
		return &t
	}()

	var pairComputation mat.Dense
	pairComputation.Add(first, second)

	result := -mat.Sum(&pairComputation) / float64(m)
	return result

}

//BackPropa back forward propagate
// before execute backpropa, the nn ForwardPropa should be executed firstly
//x, each training sample is a colum in X
//y: (k,m) ndarray
func (nn *NeuralNet) BackPropa(x mat.Matrix, yOnehot *mat.Dense) (grad []*mat.Dense) {
	//             a1       a2      a3
	//z shape:  400x5000,   25x5000,     10x5000
	//a shape: (400+1)x5000, (25+1)x5000, (10+0)x5000
	//w shape:               25x401,        10x26
	//m represent sample No.
	xr, m := x.Dims()
	//check input shape
	layer2 := &(nn.Layers[1])
	_, l2c := layer2.theta.Dims()
	if xr != l2c-1 {
		panic("input sample'shape not fit nn")
	}

	h := (nn.Layers[nn.LayerNum-1].a)
	// Deltas store graient of each layer's Weight
	Deltas := make([]*mat.Dense, nn.LayerNum)
	for i := 1; i < nn.LayerNum; i++ {
		r, c := nn.Layers[i].theta.Dims()
		Deltas[i] = mat.NewDense(r, c, nil)
	}

	// for each ith training sample
	for i := 0; i < m; i++ {

		deltas := make([][]float64, nn.LayerNum) //δ

		//back propagate each layer
		for j := nn.LayerNum - 1; j > 0; j-- { //layer-1 is input, dont need update weight
			curr := &(nn.Layers[j])

			zt := vectorToSlice(curr.z.ColView(i))
			ztLen := len(zt)
			at := vectorToSlice(curr.a.ColView(i))
			atLen := len(at) // ztlen==ztlen+1
			//at := vectorToSlice(curr.a, i)
			yt := vectorToSlice(yOnehot.ColView(i))
			ytLen := len(yt)
			ht := vectorToSlice(h.ColView(i))

			//is L
			if j == nn.LayerNum-1 {
				dts := make([]float64, ytLen)
				floats.SubTo(dts, ht, yt)
				deltas[j] = dts
			} else {
				//δ^{l} := sigmoidPrime(z) ⊙ ( (W^{l+1}).T * (δ^{l}) )
				deltas[j] = func() []float64 {
					dts := make([]float64, atLen)
					dts[0] = 1 //for bias
					for i := 0; i < ztLen; i++ {
						dts[i+1] = sigmoidPrime(zt[i])
					}
					var second mat.Dense
					second.Mul(nn.Layers[j+1].theta.T(), mat.NewVecDense(len(deltas[j+1]), deltas[j+1]))

					floats.Mul(dts, second.RawMatrix().Data)

					return dts[1:]
				}()
			}

			// accumulate each layer's
			Deltas[j] = func() *mat.Dense {
				//Δ:= (δ^{l}) * ( a^{l-1} ).T
				var DeltaTemp, r mat.Dense
				DeltaTemp.Clone(Deltas[j])

				d := mat.NewVecDense(len(deltas[j]), deltas[j])
				previousLayerA := nn.Layers[j-1].a.ColView(i).T()
				r.Mul(d, previousLayerA)

				DeltaTemp.Add(&DeltaTemp, &r)
				return &DeltaTemp
			}()
		}
	}

	//
	for i := 1; i < nn.LayerNum; i++ {
		Deltas[i].Apply(
			func(i, j int, v float64) float64 {
				return v / float64(m)
			}, Deltas[i])
	}

	grad = Deltas
	return

}

// If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001)
func (nn *NeuralNet) GradientChecking(epsilon float64) {

}

// def gradient_checking(theta, X, y, epsilon, regularized=False):
// a_numeric_grad:= func(plus, minus,):
//     """calculate a partial gradient with respect to 1 theta"""
//         return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

//     theta_matrix = expand_array(theta)  # expand to (10285, 10285)
//     epsilon_matrix = np.identity(len(theta)) * epsilon

//     plus_matrix = theta_matrix + epsilon_matrix
//     minus_matrix = theta_matrix - epsilon_matrix

//     # calculate numerical gradient with respect to all theta
//     numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
//                                     for i in range(len(theta))])

//     # analytical grad will depend on if you want it to be regularized or not
//     analytic_grad = regularized_gradient(theta, X, y) if regularized else gradient(theta, X, y)

//     # If you have a correct implementation, and assuming you used EPSILON = 0.0001
//     # the diff below should be less than 1e-9
//     # this is how original matlab code do gradient checking
//     diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

//     print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
