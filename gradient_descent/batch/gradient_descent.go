package batch

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func mainm() {
	Xd, Yd := loadData("ex1data1.txt")

	r, c := len(Xd), len(Xd[0])
	data := []float64{}
	for _, j := range Xd {
		data = append(data, j...)
	}
	X := mat.NewDense(r, c, data)
	y := mat.NewDense(r, 1, Yd)
	theta := mat.NewDense(1, r, nil)

	gradientDescent(X, y, theta, 0.01 /* alpha */, 1000 /*  inters int */)

}

// computeCost compute cost for [X,y]
// X shape is r x c, thea shape is 1 x c, y shape is r x 1
// the algorithm is same as below python function
// ```python
// def computeCost(X, y, theta):
//     inner = np.power(((X * theta.T) - y), 2)
//     return np.sum(inner) / (2 * len(X))
// ```
func computeCost(X, y, theta *mat.Dense) float64 {
	xr, xc := X.Dims()
	yr, yc := y.Dims()
	tr, tc := theta.Dims()

	if xr != yr || xc != tc || tr != 1 || yc != 1 {
		panic(mat.ErrShape)
	}

	var _error /*, inner */ mat.Dense

	_error.Mul(X, theta.T())
	_error.Sub(&_error, y)

	v := _error.ColView(0)
	sum := mat.Dot(v, v)
	sum = sum / (2 * float64(v.Len()))
	return sum
}

// ```python
// def gradientDescent(X, y, theta, alpha, iters):
//     temp = np.matrix(np.zeros(theta.shape))
//     parameters = int(theta.ravel().shape[1])
//     cost = np.zeros(iters)

//     for i in range(iters):
//         error = (X * theta.T) - y

//         for j in range(parameters):
//             term = np.multiply(error, X[:,j])
//             temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

//         theta = temp
//         cost[i] = computeCost(X, y, theta)

//     return theta, cost
// ```

// X shape is r x c, theta shape is 1 x c, y shape is r x 1

func gradientDescent(X, y, theta *mat.Dense, alpha float64, inters int) (otheta *mat.Dense, cost []float64) {
	var _error /* , term */ mat.Dense

	tr, tc := theta.Dims()
	temp := mat.NewDense(tr, tc, nil)
	_, parameters := theta.Dims()
	cost = make([]float64, inters)

	for i := 0; i < inters; i++ {
		_error.Mul(X, theta.T()) //_error shape will be r x 1
		_error.Sub(&_error, y)

		for j := 0; j < parameters; j++ {
			v := _error.ColView(0)
			sum := mat.Dot(v, v)
			sum = sum / (float64(v.Len()))
			temp.Set(0, j, theta.At(0, j)-(float64(alpha)*sum))

			fmt.Println("sum:", sum)
		}

		theta = temp
		cost[i] = computeCost(X, y, theta)
	}

	return theta, cost
}

func Sum(m *mat.Dense) float64 {
	r, c := m.Dims()

	var sum float64
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum = sum + m.At(i, j)
		}
	}
	return sum
}

// func shuffle(X [][]float64, rnd func(n int) int) [][]float64 {

// 	newDs := make([][]float64, len(X))
// 	copy(newDs, X)

// 	for i, _ := range X {
// 		j := i + rnd(len(X)-i)
// 		newDs[i], newDs[j] = newDs[j], newDs[i]
// 	}
// 	return newDs
// }
