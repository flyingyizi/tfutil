package logicregression

import (
	"math"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

//ComputeCost , computer logistic regression cost
//$$J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \left[  -y^{(i)} \times log(h_{\theta}(x^{(i)})) - (1-y^{(i)}) \times log(1- h_{\theta}(x^{(i)}))  \right] $$
func ComputeCost(h Fhyphothesis, X *mat.Dense, y *mat.VecDense, theta mat.Vector) float64 {

	// $log(h_{\theta}(x^{(i)})$
	H := func() *mat.Dense {
		a := h(X, theta)
		return a
	}()

	//  $-y^{(i)} \times log(h_{\theta}(x^{(i)}))$
	first := func() *mat.Dense {
		var logH mat.Dense

		logH.Apply(
			func(i, j int, v float64) float64 {
				return math.Log(v)
			}, H)

		var a mat.Dense
		a.MulElem(y, &logH)
		a.Apply(
			func(i, j int, v float64) float64 {
				return -v
			}, &a)
		return &a
	}()

	// $(1-y^{(i)}) \times log(1- h_{\theta}(x^{(i)})$
	second := func() *mat.Dense {
		var a mat.Dense

		a.Apply(
			func(i, j int, v float64) float64 {
				return math.Log(1 - v)
			}, H)

		var b mat.Dense
		b.MulElem(y, &a)

		a.Sub(&a, &b)
		return &a
	}()

	// $\frac{1}{m} \sum_{i=1}^{m} \left[ first - second \right]$
	result := func() float64 {
		first.Sub(first, second)

		m, _ := first.Dims()
		sum := floats.Sum(first.RawMatrix().Data)
		sum = sum / float64(m)

		return sum
	}()

	return result
}

//Fhyphothesis  hyphothesis function type
type Fhyphothesis func(X mat.Matrix, theta mat.Matrix) *mat.Dense

//LogicHyphothesis output Probability of $h_\theta(x)=P(y=1|x;\theta)$
// the theta shape must be any by 1
func LogicHyphothesis(X mat.Matrix, theta mat.Matrix) *mat.Dense {

	if _, thetac := theta.Dims(); thetac != 1 {
		panic("wrong theta size")
	}

	var h mat.Dense
	h.Mul(X, theta)
	h.Apply(
		func(i, j int, v float64) float64 {
			t := 1.0 / (1.0 + math.Exp(-v))
			return t
		}, &h)
	return &h
}

// BGD batch gradient descent for logistic regression
func BGD(h Fhyphothesis, X *mat.Dense, y *mat.VecDense, theta mat.Vector, alpha float64, inters int, outputCost bool) (otheta, cost []float64) {
	xr, xc := X.Dims()
	yr, _ := y.Dims()
	tr := theta.Len()

	if xr != yr || xc != tr {
		panic(mat.ErrShape)
	}

	parameters := tr
	if outputCost {
		cost = make([]float64, inters)
	}
	for i := 0; i < inters; i++ {
		_error := h(X, theta)
		_error.Sub(_error, y)

		temp := mat.NewVecDense(tr, nil)
		for j := 0; j < parameters; j++ {
			sum := mat.Dot(_error.ColView(0), X.ColView(j))
			sum = (float64(alpha) / float64(xr)) * sum
			temp.SetVec(j, theta.AtVec(j)-sum)
		}

		theta = temp
		if outputCost {
			cost[i] = ComputeCost(LogicHyphothesis, X, y, theta)
		}
	}

	otheta = make([]float64, theta.Len())
	for i := 0; i < theta.Len(); i++ {
		otheta[i] = theta.AtVec(i)
	}
	return otheta, cost
}

//LogisticPredict output y, if $h_\theta(x)$ >=0.5, the y will  be 1,
//else will be 0, the $h_\theta(x)$ is $h_\theta(x)=P(y=1|x;\theta)$
func LogisticPredict(X *mat.Dense, theta mat.Vector) []float64 {
	p := LogicHyphothesis(X, theta)
	pr, _ := p.Dims()

	newout := make([]float64, pr)

	p.Apply(
		func(i, j int, v float64) float64 {
			if v >= 0.5 {
				return 1
			}
			return 0

		}, p)

	copy(newout, p.RawMatrix().Data)

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
