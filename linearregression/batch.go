package linearregression

import (
	"gonum.org/v1/gonum/mat"
)

// ComputeCost compute cost for [X,y]
// X shape is r x c, thea shape is c x 1, y shape is r x 1
// the algorithm is same as below python function
// ```python
// def computeCost(X, y, theta):
//     inner = np.power(((X * theta.T) - y), 2)
//     return np.sum(inner) / (2 * len(X))
// ```
func ComputeCost(X *mat.Dense, y, theta mat.Vector) float64 {
	xr, xc := X.Dims()
	yr, yc := y.Dims()
	tr := theta.Len()

	if xr != yr || xc != tr || yc != 1 {
		panic(mat.ErrShape)
	}

	var _error /*, inner */ mat.VecDense

	_error.MulVec(X, theta)
	_error.SubVec(&_error, y)

	sum := mat.Dot(&_error, &_error)
	sum = sum / (2 * float64(_error.Len()))
	return sum
}

// ```python
// # https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes
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
//         cost[i] = ComputeCost(X, y, theta)

//     return theta, cost
// ```

// X shape is r x c, theta shape is c x 1, y shape is r x 1
// GradientDescent
func GradientDescent(X *mat.Dense, y, theta mat.Vector, alpha float64, inters int) (otheta, cost []float64) {
	var _error /* , term */ mat.VecDense

	xr, xc := X.Dims()
	yr, _ := y.Dims()
	tr := theta.Len()

	if xr != yr || xc != tr {
		panic(mat.ErrShape)
	}

	parameters := tr
	cost = make([]float64, inters)

	for i := 0; i < inters; i++ {
		_error.MulVec(X, theta)
		_error.SubVec(&_error, y)

		temp := mat.NewVecDense(tr, nil)
		for j := 0; j < parameters; j++ {
			sum := mat.Dot(&_error, X.ColView(j))
			sum = (float64(alpha) / float64(xr)) * sum
			temp.SetVec(j, theta.AtVec(j)-sum)
		}

		theta = temp
		cost[i] = ComputeCost(X, y, theta)
	}

	otheta = make([]float64, theta.Len())
	for i := 0; i < theta.Len(); i++ {
		otheta[i] = theta.AtVec(i)
	}
	return otheta, cost
}

// func Sum(m *mat.Dense) float64 {
// 	r, c := m.Dims()

// 	var sum float64
// 	for i := 0; i < r; i++ {
// 		for j := 0; j < c; j++ {
// 			sum = sum + m.At(i, j)
// 		}
// 	}
// 	return sum
// }

// func shuffle(X [][]float64, rnd func(n int) int) [][]float64 {

// 	newDs := make([][]float64, len(X))
// 	copy(newDs, X)

// 	for i, _ := range X {
// 		j := i + rnd(len(X)-i)
// 		newDs[i], newDs[j] = newDs[j], newDs[i]
// 	}
// 	return newDs
// }
