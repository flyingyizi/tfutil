package iforest_test

import (
	"path"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/df"
	"github.com/flyingyizi/tfutil/unsupervisedLearning/iforest"
	"gonum.org/v1/gonum/mat"
)

func ExampleIforest() {
	//load training X
	filename := "ex7data2.txt"
	X := mat.NewDense(df.CsvToArray(path.Join("testdata", "X"+filename)))
	m, _ := X.Dims()

	//assign original scatter plot data
	origScatt := &tfutil.ScatterData{}
	xs, ys, zs := make([]float64, m), make([]float64, m), make([]float64, m)
	for i := 0; i < m; i++ {
		xs[i], ys[i] = X.At(i, 0), X.At(i, 1)
	}
	origScatt.Add("orig", xs, ys, zs)

	//assign original scatter plot data
	// clusterScatt := &tfutil.ScatterData{}
	// for i := 0; i < m; i++ {
	// 	xs[i], ys[i], zs[i] = X.At(i, 0), X.At(i, 1), float64(idx[i])
	// }
	// clusterScatt.Add("cluster", xs, ys, zs)

	// tfutil.SaveScatters(filename, origScatt, clusterScatt)
	// fmt.Println("")

	//model initialization
	forest := iforest.NewForest(0.01, iforest.NbTrees(100), iforest.SubsamplingSize(256))

	//training stage - creating trees
	forest.Train(X.T())

	//testing stage - finding anomalies
	//Test or TestParaller can be used, concurrent version needs one additional
	// parameter
	forest.Test(X.T())
	//output:
	//
}

// input parameters
// treesNumber := 100
// subsampleSize := 256
// outliersRatio := 0.01
// routinesNumber := 10

// forest.TestParallel(inputData, routinesNumber)

//after testing it is possible to access anomaly scores, anomaly bound
// and labels for the input dataset
// threshold := forest.AnomalyBound
// anomalyScores := forest.AnomalyScores
// labelsTest := forest.Labels

//to get information about new instances pass them to the Predict function
// to speed up computation use concurrent version of Predict
// var newData [][]float64
// newData = loadData("someNewInstances")
// labels, scores := forest.Predict(newData)

// func ExampleIforest() {

// 	// input data must be loaded into two dimensional array of the type float64
// 	// please note: loadData() is some custom function - not included in the
// 	// library
// 	var inputData [][]float64
// 	inputData = loadData("filename")

// 	// input parameters
// 	treesNumber := 100
// 	subsampleSize := 256
// 	outliersRatio := 0.01
// 	routinesNumber := 10

// 	//model initialization
// 	forest := iforest.NewForest(treesNumber, subsampleSize, outliers)

// 	//training stage - creating trees
// 	forest.Train(inputData)

// 	//testing stage - finding anomalies
// 	//Test or TestParaller can be used, concurrent version needs one additional
// 	// parameter
// 	forest.Test(inputData)
// 	forest.TestParallel(inputData, routinesNumber)

// 	//after testing it is possible to access anomaly scores, anomaly bound
// 	// and labels for the input dataset
// 	threshold := forest.AnomalyBound
// 	anomalyScores := forest.AnomalyScores
// 	labelsTest := forest.Labels

// 	//to get information about new instances pass them to the Predict function
// 	// to speed up computation use concurrent version of Predict
// 	var newData [][]float64
// 	newData = loadData("someNewInstances")
// 	labels, scores := forest.Predict(newData)

// }
