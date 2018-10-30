package logicregression

import (
	"math"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

// def sigmoid(z):
// 	return 1 / (1 + np.exp(-z))

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// def cost(theta, X, y):
//     theta = np.matrix(theta)
//     X = np.matrix(X)
//     y = np.matrix(y)
//     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
//     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
//     return np.sum(first - second) / (len(X))

func ComputeCost(X *mat.Dense, y *mat.VecDense, theta mat.Vector) float64 {
	xr, xc := X.Dims()
	yr, yc := y.Dims()
	tr := theta.Len()

	if xr != yr || xc != tr || yc != 1 {
		panic(mat.ErrShape)
	}

	var xtheta mat.VecDense
	xtheta.MulVec(X, theta)
	array, t1, t2 := make([]float64, xtheta.Len()), make([]float64, xtheta.Len()), make([]float64, xtheta.Len())
	copy(array, xtheta.RawVector().Data)
	for i := 0; i < xr; i++ {
		array[i] = math.Log(sigmoid(xtheta.AtVec(i)))
	}
	floats.MulTo(t1, y.RawVector().Data, array)
	floats.Scale(-1, t1)

	//calc second
	for i := 0; i < xr; i++ {
		array[i] = math.Log(1 - sigmoid(xtheta.AtVec(i)))
	}
	ones := func() []float64 {
		data := make([]float64, yr)
		for i := 0; i < yr; i++ {
			data[i] = 1
		}
		return data
	}()
	floats.Sub(ones, y.RawVector().Data)
	floats.MulTo(t2, ones, array)

	//     return np.sum(first - second) / (len(X))
	floats.Sub(t1, t2)
	return floats.Sum(t1) / float64(yr)
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
func GradientDescent(X *mat.Dense, y *mat.VecDense, theta mat.Vector, alpha float64, inters int) (otheta, cost []float64) {
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

// def predict(theta, X):
//     probability = sigmoid(X * theta.T)
//     return [1 if x >= 0.5 else 0 for x in probability]
