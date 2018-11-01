package logicregression

import (
	"math"

	"gonum.org/v1/gonum/floats"

	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"

	"gonum.org/v1/gonum/stat"
)

// def cost(theta, X, y):
//     theta = np.matrix(theta)
//     X = np.matrix(X)
//     y = np.matrix(y)
//     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
//     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
//     return np.sum(first - second) / (len(X))

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

// CsvToDense,
// csv文件中的列数据在输出到orig时,orgin[i]代表原始数据的第i列
// 当normalize为true时，代表输出的X，Y经过正规化处理; 否则数据为向量化后的原始数据
// 由于logistic regression的y必须是0/1，因此无论是否正规化数据，该函数都不会变更y的值
func CsvToDense(filename string, normalize bool) (X *mat.Dense, Y *mat.VecDense, orign [][]float64) {
	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)

	reader.Comment = '#' //可以设置读入文件中的注释符
	reader.Comma = ','   //默认是逗号，也可以自己设置

	firstRecord, err := reader.Read()
	if err == io.EOF {
		return
	} else if err != nil {
		fmt.Println("Error:", err)
		return
	}
	cols := len(firstRecord)
	data := make([][]float64, cols+1)

	row := 0
	for i := 0; i < cols; i++ {
		data[i+1] = make([]float64, 0)
		if value, err := strconv.ParseFloat(firstRecord[i], 64); err == nil {
			data[i+1] = append(data[i+1], value)
		} else {
			return
		}
	}
	row++

	for {
		// continue scan
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println("Error:", err)
			return
		}
		for i, j := range record {
			if value, err := strconv.ParseFloat(j, 64); err == nil {
				data[i+1] = append(data[i+1], value)
			} else {
				return
			}
		}
		row++
	}

	data[0] = make([]float64, row)
	for i := 0; i < row; i++ {
		data[0][i] = 1
	}

	//assign orig data
	orign = make([][]float64, cols)
	for i := 0; i < cols; i++ {
		orign[i] = make([]float64, row)
		copy(orign[i], data[i+1])
	}

	if normalize {
		// ones line & y line dont need feature normalize
		for i := 1; i < cols; i++ {
			m := stat.Mean(data[i], nil)
			s := stat.StdDev(data[i], nil)
			floats.AddConst(-1*m, data[i])
			if s != 0.0 {
				floats.Scale(1/s, data[i])
			}
		}
		X = mat.NewDense(row, cols, nil)
		for i := 0; i < cols; i++ {
			X.SetCol(i, data[i])
		}
	} else {
		X = mat.NewDense(row, cols, nil)
		for i := 0; i < cols; i++ {
			X.SetCol(i, data[i])
		}
	}

	Y = mat.NewVecDense(row, data[cols])

	return

}
