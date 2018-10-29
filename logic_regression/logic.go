package logic_regression

import (
	"math"
)

// def sigmoid(z):
// 	return 1 / (1 + np.exp(-z))

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}
