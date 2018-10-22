package batch

import (
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

// ```python
// def computeCost(X, y, theta):
//     inner = np.power(((X * theta.T) - y), 2)
//     return np.sum(inner) / (2 * len(X))
// ```
func computeCost(X, y, theta *mat.Dense) float64 {
	var _error, inner mat.Dense
	r, _ := X.Dims()

	_error.MulElem(X, theta.T())
	_error.Sub(&_error, y)
	inner.MulElem(&_error, &_error)

	return Sum(&inner) / (2 * float64(r))
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
func gradientDescent(X, y, theta *mat.Dense, alpha float64, inters int) (otheta *mat.Dense, cost []float64) {
	var _error, term mat.Dense

	r, c := theta.Dims()
	temp := mat.NewDense(r, c, nil)
	_, parameters := theta.Dims()
	cost = make([]float64, inters)

	for i := 0; i < inters; i++ {
		_error.MulElem(X, theta.T())
		_error.Sub(&_error, y)

		for j := 0; j < parameters; j++ {
			term.Mul(&_error, X.ColView(j).T())
			temp.Set(0, j, theta.At(0, j)-((float64(alpha)/float64(r))*Sum(&term)))
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
