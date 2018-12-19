package iforest_test

import (
	"fmt"
	"path"
	"time"

	"gonum.org/v1/gonum/floats"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/df"
	"github.com/flyingyizi/tfutil/unsupervisedLearning/iforest"
	"gonum.org/v1/gonum/mat"
)

func ExampleForest() {
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

	//model initialization
	forest := iforest.NewForest(
		iforest.AnomalyRatio(0.0001),
		iforest.NbTrees(100),
		iforest.SubsamplingSize(256))

	//training stage - creating trees
	forest.Train(X.T())

	//testing stage - finding anomalies
	//Test or TestParaller can be used, concurrent version needs one additional
	// parameter
	forest.Test(X.T())
	// labels, _, _ := forest.Predict(X.T())
	scores := forest.AnomalyScores

	for i := 0; i < len(scores); i++ {
		if scores[i] < 0.5 {
			fmt.Println("score large than 0.5 has:", i, " total:", m)
			break
		}
	}
	labelsTest := forest.Labels

	//assign labeled scatter plot data
	clusterScatt := &tfutil.ScatterData{}
	for i := 0; i < m; i++ {
		xs[i], ys[i], zs[i] = X.At(i, 0), X.At(i, 1), float64(labelsTest[i])
	}
	clusterScatt.Add("cluster", xs, ys, zs)

	tfutil.SaveScatters(filename, origScatt, clusterScatt)

	tfutil.SaveBoxPlot("box-"+filename, forest.AnomalyScores)
	fmt.Println("")
	//output:
	//
}

func ExampleForest_abcd() {

	//load training X
	filename := "abcd.txt"
	temp := mat.NewDense(df.CsvToArray(path.Join("testdata", filename)))
	m, _ := temp.Dims()

	//info: time, cpuPercent , memUsedPercent, net recB,sendB, num of conn
	//get time
	locTime, cpuPercent, memPercent := mat.Col(nil, 0, temp), mat.Col(nil, 1, temp), mat.Col(nil, 2, temp)
	netRec, netSent, numc := mat.Col(nil, 3, temp), mat.Col(nil, 4, temp), mat.Col(nil, 5, temp)
	floats.Div(netRec, numc)
	floats.Div(netSent, numc)

	xx := mat.NewDense(m, 4, nil)
	xx.SetCol(0, df.FeatureScaling(cpuPercent...))
	xx.SetCol(1, df.FeatureScaling(memPercent...))
	xx.SetCol(2, df.FeatureScaling(numc...))

	//from 3xm to 2xm
	k := 2
	p := df.NewPCA(k)
	tra, _ := p.FitTransform(xx.T())
	//fmt.Printf("%0.4f", mat.Formatted(tra, mat.Prefix(""), mat.Squeeze()))

	//assign original scatter plot data
	origScatt := &tfutil.ScatterData{}
	xs, ys, zs := make([]float64, m), make([]float64, m), make([]float64, m)
	for i := 0; i < m; i++ {
		xs[i], ys[i] = tra.At(0, i), tra.At(1, i)
	}
	origScatt.Add("orig", xs, ys, zs)

	//model initialization
	forest := iforest.NewForest(
		iforest.AnomalyRatio(0.0001),
		iforest.NbTrees(100),
		iforest.SubsamplingSize(256))

	//training and testing stage - creating trees, finding anomalies
	// forest.Train(xx.T())
	// forest.Test(xx.T())
	forest.Train(tra)
	forest.Test(tra)

	// scores := make([]float64, len(forest.AnomalyScores))
	// copy(scores, forest.AnomalyScores)
	//  sorts given in descending order.
	// sort.Slice(scores, func(i, j int) bool {
	// 	return scores[i] > scores[j]
	// })
	// for i := 0; i < len(scores); i++ {
	// 	if scores[i] < 0.75 {
	// 		fmt.Println("score lager than 0.5 has:", i, " total:", m, "  AnomalyBound:", forest.AnomalyBound)
	// 		break
	// 	}
	// }
	labelsTest := forest.Labels
	j := 0
	for i := 0; i < m; i++ {
		if labelsTest[i] == 1 {
			j++
			fmt.Printf("%v:%v\n", time.Unix(0, int64(locTime[i])), forest.AnomalyScores[i])
		}
	}
	fmt.Printf("totoal anomaly:%v", j)

	//assign labeled scatter plot data
	clusterScatt := &tfutil.ScatterData{}
	for i := 0; i < m; i++ {
		xs[i], ys[i], zs[i] = tra.At(0, i), tra.At(1, i), float64(labelsTest[i])
	}
	clusterScatt.Add("cluster", xs, ys, zs)

	tfutil.SaveScatters(filename, origScatt, clusterScatt)

	tfutil.SaveBoxPlot("box-"+filename, forest.AnomalyScores)

	tfutil.SaveHistograms("hist-"+filename, forest.AnomalyScores)
	forest.Save("testdata\\saved.txt.json")
	fmt.Println("")
	//output:
	//
}
