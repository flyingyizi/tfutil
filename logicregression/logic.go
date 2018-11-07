package logicregression

import (
	"math"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

//ComputeCost , with special theta to compute the cost
//: $$J(\theta)=\frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2 $$
func ComputeCost(h Fhyphothesis, X *mat.Dense, y *mat.VecDense, theta mat.Vector) float64 {
	xr, xc := X.Dims()
	yr, yc := y.Dims()
	tr := theta.Len()

	if xr != yr || xc != tr || yc != 1 {
		panic(mat.ErrShape)
	}

	var xtheta mat.VecDense
	xtheta.MulVec(X, theta)

	hyph := h(X, theta)

	array, t1, t2 := make([]float64, hyph.Len()), make([]float64, xtheta.Len()), make([]float64, xtheta.Len())
	copy(array, xtheta.RawVector().Data)
	for i := 0; i < xr; i++ {
		array[i] = math.Log(hyph.AtVec(i))
	}
	floats.MulTo(t1, y.RawVector().Data, array)
	floats.Scale(-1, t1)

	//calc second
	for i := 0; i < xr; i++ {
		array[i] = math.Log(1 - hyph.AtVec(i))
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

type Fhyphothesis func(X mat.Matrix, theta mat.Vector) *mat.VecDense

func linearHyphothesis(X mat.Matrix, theta mat.Vector) *mat.VecDense {
	var h mat.VecDense
	h.MulVec(X, theta)
	return &h
}

func LogicHyphothesis(X mat.Matrix, theta mat.Vector) *mat.VecDense {
	var h mat.VecDense
	h.MulVec(X, theta)

	for i := 0; i < h.Len(); i++ {
		t := 1 / (1 + math.Exp(-h.AtVec(i)))
		h.SetVec(i, t)
	}

	return &h
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
func GradientDescent(h Fhyphothesis, X *mat.Dense, y *mat.VecDense, theta mat.Vector, alpha float64, inters int) (otheta, cost []float64) {
	xr, xc := X.Dims()
	yr, _ := y.Dims()
	tr := theta.Len()

	if xr != yr || xc != tr {
		panic(mat.ErrShape)
	}

	parameters := tr
	cost = make([]float64, inters)

	for i := 0; i < inters; i++ {
		_error := h(X, theta)
		_error.SubVec(_error, y)

		temp := mat.NewVecDense(tr, nil)
		for j := 0; j < parameters; j++ {
			sum := mat.Dot(_error, X.ColView(j))
			sum = (float64(alpha) / float64(xr)) * sum
			temp.SetVec(j, theta.AtVec(j)-sum)
		}

		theta = temp
		cost[i] = ComputeCost(LogicHyphothesis, X, y, theta)
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
func Predict(X *mat.Dense, theta mat.Vector) []int {
	p := LogicHyphothesis(X, theta)
	newout := make([]int, p.Len())

	for i := 0; i < p.Len(); i++ {
		if p.AtVec(i) >= 0.5 {
			newout[i] = 1
		} else {
			newout[i] = 0
		}
	}
	return newout
}

// def predict_all(X, all_theta):
//     rows = X.shape[0]
//     params = X.shape[1]
//     num_labels = all_theta.shape[0]

//     # same as before, insert ones to match the shape
//     X = np.insert(X, 0, values=np.ones(rows), axis=1)

//     # convert to matrices
//     X = np.matrix(X)
//     all_theta = np.matrix(all_theta)

//     # compute the class probability for each class on each training instance
//     h = sigmoid(X * all_theta.T)

//     # create array of the index with the maximum probability
//     h_argmax = np.argmax(h, axis=1)

//     # because our array was zero-indexed we need to add one for the true label prediction
//     h_argmax = h_argmax + 1

//     return h_argmax

// y_pred = predict_all(data['X'], all_theta)
// correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
// accuracy = (sum(map(int, correct)) / float(len(correct)))
// print ('accuracy = {0}%'.format(accuracy * 100))
