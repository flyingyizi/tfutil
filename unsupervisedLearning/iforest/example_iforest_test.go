package iforest_test

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
