package iforest

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// Euler is an Euler's constant as described in algorithm specification
const Euler float64 = 0.5772156649

type kv struct {
	Key   int
	Value float64
}

func computeC(n float64) float64 {
	return 2*(math.Log(n-1)+Euler) - ((2 * (n - 1)) / n)
}

type optionalAttr struct {
	NbTrees         int
	SubsamplingSize int
}

// NewForestAttr is an optional argument to create Forest.
type NewForestAttr func(*optionalAttr)

// NbTrees sets the optional number of tree attribute to value.
//
// value: number of tree to be set. If not specified, defaults to 100
func NbTrees(value int) NewForestAttr {
	return func(m *optionalAttr) {
		m.NbTrees = value
	}
}

// SubsamplingSize sets the optional sub-sample size attribute to value.
//
// value: number of tree to be set. If not specified, defaults to 256
func SubsamplingSize(value int) NewForestAttr {
	return func(m *optionalAttr) {
		m.SubsamplingSize = value
	}
}

// Forest is a base structure for Isolation Forest algorithm. It holds algorithm
// parameters like number of trees, subsampling size, anomalies ratio and
// collection of created trees.
type Forest struct {
	Option optionalAttr

	Trees []Tree

	HeightLimit   int
	AnomalyScores map[int]float64
	AnomalyBound  float64
	AnomalyRatio  float64
	Labels        []int
	Trained       bool
	Tested        bool
}

//NewForest initializes Forest structure.
func NewForest(anomalyRatio float64, optional ...NewForestAttr) *Forest {
	//default
	attrs := optionalAttr{NbTrees: 100, SubsamplingSize: 256}
	// customize
	for _, a := range optional {
		a(&attrs)
	}
	f := &Forest{Option: attrs}

	f.HeightLimit = int(math.Ceil(math.Log2(float64(attrs.SubsamplingSize))))
	f.Trees = make([]Tree, attrs.NbTrees)
	f.AnomalyScores = make(map[int]float64)

	f.AnomalyRatio = anomalyRatio
	return f
}

// Train creates the collection of trees in the forest. This is the training
// stage of the algorithm.
func (f *Forest) Train(X mat.Matrix) {
	_, xc := X.Dims()
	// using golang map's feature (range visit is radom visiting)
	//to radom select the sub-sample for the tree
	mx := make(map[int]float32, xc)
	for i := 0; i < xc; i++ {
		mx[i] = 0.0 //we only need key
	}

	for i := 0; i < f.Option.NbTrees; i++ {
		subsamplesIndicies := f.createSubsamplesWithoutReplacement(mx)
		f.Trees[i] = Tree{}
		f.Trees[i].BuildTree(X, subsamplesIndicies, f.HeightLimit)
	}
	f.Trained = true
}

// Test is the algorithm "Evaluating Stage". It computes anomaly scores for the
// dataset (should be used with the same set as in training) and chooses anomaly
// score that will be the bound for detecting anomalies.
func (f *Forest) Test(X mat.Matrix) error {
	//m represent number of instance
	_, m := X.Dims()

	if !f.Trained {
		return errors.New("cannot start testing phase - model has not been trained yet")
	}

	//calc score for each instance

	// • (a) if instances return s very close to 1, then they are
	// definitely anomalies,
	// • (b) if instances have s much smaller than 0.5, then they
	// are quite safe to be regarded as normal instances, and
	// • (c) if all the instances return s ≈ 0.5, then the entire
	// sample does not really have any distinct anomaly.

	cn := computeC(float64(f.Option.SubsamplingSize))
	for i := 0; i < m; i++ {
		instance := mat.Col(nil, i, X)

		sum := 0.0
		for j := 0; j < len(f.Trees); j++ {
			path := f.Trees[j].PathLength(instance)
			sum += path
		}
		// $$s(x, n) = 2^{- \frac{E(h(x))}{c(n)}  } ,$$
		average := sum / float64(len(f.Trees))
		s := math.Pow(2, (-average / cn))
		//s= 0.5 - s
		f.AnomalyScores[i] = s
	}

	sorted := sortMap(f.AnomalyScores)
	anomFloor := int(math.Floor(f.AnomalyRatio * float64(m)))
	anomCeil := int(math.Ceil(f.AnomalyRatio * float64(m)))
	f.AnomalyBound = (sorted[anomFloor].Value + sorted[anomCeil].Value) / 2

	f.Labels = make([]int, m)
	for i := 0; i < m; i++ {
		if f.AnomalyScores[i] < f.AnomalyBound {
			f.Labels[i] = 1
		} else {
			f.Labels[i] = 0
		}
	}

	f.Tested = true

	return nil

}

// Predict computes anomaly scores for given dataset and classifies each vector
// as 'normal' or 'anomaly'.
func (f *Forest) Predict(X mat.Matrix) ([]int, []float64, error) {
	_, m := X.Dims()

	if !f.Trained {
		return nil, nil, errors.New("cannot predict - model has not been trained yet")
	}
	if !f.Tested {
		return nil, nil, errors.New("cannot predict - model has not been tested yet")
	}

	labels := make([]int, m)
	scores := make([]float64, m)

	cn := computeC(float64(f.Option.SubsamplingSize))
	for i := 0; i < m; i++ {
		instance := mat.Col(nil, i, X)

		var sumPathLength float64
		for j := 0; j < len(f.Trees); j++ {
			path := f.Trees[j].PathLength(instance)
			sumPathLength += path
		}
		// $$s(x, n) = 2^{- \frac{E(h(x))}{c(n)}  } ,$$
		average := sumPathLength / float64(len(f.Trees))
		s := math.Pow(2, (-average / cn))
		//s= 0.5 - s

		scores[i] = s
	}

	for i := 0; i < m; i++ {
		if scores[i] < f.AnomalyBound {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	return labels, scores, nil

}

// TestParallel does the same as Test but using multiple go routines
func (f *Forest) TestParallel(X [][]float64, routinesNumber int) error {

	if !f.Trained {
		return errors.New("cannot start testing phase - model has not been trained yet")
	}

	if routinesNumber > len(X) || routinesNumber == 0 {
		return errors.New("number of routines cannot be bigger than nubmer of vectors or equal to 0")
	}
	var CAnomalyScores sync.Map
	var wg sync.WaitGroup
	wg.Add(routinesNumber)
	vectorsPerRoutine := int(len(X) / routinesNumber)

	for j := 0; j < routinesNumber; j++ {
		go f.computeAnomalies(X, j*vectorsPerRoutine, j*vectorsPerRoutine+vectorsPerRoutine, &wg, &CAnomalyScores)
	}
	wg.Wait()

	CAnomalyScores.Range(func(key, value interface{}) bool {
		f.AnomalyScores[key.(int)] = value.(float64)
		return true
	})

	sorted := sortMap(f.AnomalyScores)
	anomFloor := int(math.Floor(f.AnomalyRatio * float64(len(X))))
	anomCeil := int(math.Ceil(f.AnomalyRatio * float64(len(X))))
	f.AnomalyBound = (sorted[anomFloor].Value + sorted[anomCeil].Value) / 2

	f.Labels = make([]int, len(X))
	for i := 0; i < len(X); i++ {
		if f.AnomalyScores[i] < f.AnomalyBound {
			f.Labels[i] = 1
		} else {
			f.Labels[i] = 0
		}
	}
	return nil
}

// PredictParallel computes anomaly scores for given dataset and classifies each vector
// as 'normal' or 'anomaly'. Uses  multiple go routines to make computation faster.
func (f *Forest) PredictParallel(X [][]float64, routinesNumber int) ([]int, []float64, error) {

	if !f.Trained {
		return nil, nil, errors.New("cannot predict - model has not been trained yet")
	}

	if !f.Tested {
		return nil, nil, errors.New("cannot predict - model has not been tested yet")
	}

	if routinesNumber > len(X) || routinesNumber == 0 {
		return nil, nil, errors.New("number of routines cannot be bigger than nubmer of vectors or equal to 0")
	}

	labels := make([]int, len(X))
	scores := make([]float64, len(X))

	var wg sync.WaitGroup
	wg.Add(routinesNumber)
	vectorsPerRoutine := int(len(X) / routinesNumber)

	cn := computeC(float64(f.Option.SubsamplingSize))
	for j := 0; j < routinesNumber; j++ {
		go func(start, stop int) {
			for i := start; i < stop; i++ {
				var sumPathLength float64
				sumPathLength = 0.0
				for j := 0; j < len(f.Trees); j++ {
					path := f.Trees[j].PathLength(X[i])
					sumPathLength += path

				}
				// $$s(x, n) = 2^{- \frac{E(h(x))}{c(n)}  } ,$$
				average := sumPathLength / float64(len(f.Trees))
				s := math.Pow(2, (-average / cn))
				//s= 0.5 - s

				scores[i] = s
			}

			wg.Done()
		}(j*vectorsPerRoutine, j*vectorsPerRoutine+vectorsPerRoutine)
	}
	wg.Wait()

	for i := 0; i < len(X); i++ {
		if scores[i] < f.AnomalyBound {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	return labels, scores, nil

}

func (f *Forest) createSubsamplesWithoutReplacement(X map[int]float32) []int {

	subsamplesIds := make([]int, f.Option.SubsamplingSize)
	currentSize := 0
	for i := range X {
		subsamplesIds[currentSize] = i
		currentSize++
		if currentSize >= f.Option.SubsamplingSize {
			return subsamplesIds
		}
	}
	return subsamplesIds
}

func (f *Forest) computeAnomalies(X [][]float64, start, stop int, wg *sync.WaitGroup, as *sync.Map) {

	cn := computeC(float64(f.Option.SubsamplingSize))

	for i := start; i < stop; i++ {
		var sumPathLength float64
		sumPathLength = 0.0
		for j := 0; j < len(f.Trees); j++ {
			path := f.Trees[j].PathLength(X[i])
			sumPathLength += path
		}

		// $$s(x, n) = 2^{- \frac{E(h(x))}{c(n)}  } ,$$
		average := sumPathLength / float64(len(f.Trees))
		s := math.Pow(2, (-average / cn))
		//s= 0.5 - s

		as.Store(i, s)
	}

	wg.Done()
}

// sortMap sorts given map in descending order.
func sortMap(m map[int]float64) []kv {
	var ss []kv
	for k, v := range m {
		ss = append(ss, kv{k, v})
	}

	sort.Slice(ss, func(i, j int) bool {
		return ss[i].Value > ss[j].Value
	})

	return ss
}

// Save saves model in the file
func (f *Forest) Save(path string) error {

	file, err := os.Create(path)
	if err == nil {
		encoder := json.NewEncoder(file)
		err = encoder.Encode(f)
	}
	file.Close()
	return err
}

// Load loads from the file
func (f *Forest) Load(path string) error {
	file, err := os.Open(path)
	if err == nil {
		f.AnomalyScores = make(map[int]float64)
		decoder := json.NewDecoder(file)
		err = decoder.Decode(&f)
	}
	file.Close()

	return err
}
