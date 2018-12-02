package iforest

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"sort"

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
	AnomalyRatio    float64
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

// AnomalyRatio sets the optional anomaly ratio attribute to value.
//
// value: anomaly ratio to be set. If not specified, defaults to 0.01
func AnomalyRatio(value float64) NewForestAttr {
	return func(m *optionalAttr) {
		m.AnomalyRatio = value
	}
}

// Forest is a base structure for Isolation Forest algorithm. It holds algorithm
// parameters like number of trees, subsampling size, anomalies ratio and
// collection of created trees.
type Forest struct {
	Option optionalAttr

	Trees []Tree

	HeightLimit   int
	AnomalyScores []float64 //instance score in descending order
	AnomalyBound  float64
	Labels        []int
	Trained       bool
	Tested        bool
}

//NewForest initializes Forest structure.
func NewForest(optional ...NewForestAttr) *Forest {
	//default
	attrs := optionalAttr{NbTrees: 100, SubsamplingSize: 256, AnomalyRatio: 0.01}
	// customize
	for _, a := range optional {
		a(&attrs)
	}
	f := &Forest{Option: attrs}

	f.HeightLimit = int(math.Ceil(math.Log2(float64(attrs.SubsamplingSize))))
	f.Trees = make([]Tree, attrs.NbTrees)

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
//
// • (a) if instances return s very close to 1, then they are definitely anomalies,
// • (b) if instances have s much smaller than 0.5, then they
// are quite safe to be regarded as normal instances, and
// • (c) if all the instances return s ≈ 0.5, then the entire
// sample does not really have any distinct anomaly.
func (f *Forest) Test(X mat.Matrix) error {
	//m represent number of instance
	_, m := X.Dims()

	if !f.Trained {
		return errors.New("cannot start testing phase - model has not been trained yet")
	}

	if len(f.AnomalyScores) != m {
		f.AnomalyScores = make([]float64, m)
	}
	if len(f.Labels) != m {
		f.Labels = make([]int, m)
	}

	//calc score for each instance
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
		//s= s- 0.5
		f.AnomalyScores[i] = s
	}

	temp := make([]float64, len(f.AnomalyScores))
	copy(temp, f.AnomalyScores)

	//  sorts given in descending order.
	sort.Slice(temp, func(i, j int) bool {
		return temp[i] > temp[j]
	})

	anomFloor := int(math.Floor(f.Option.AnomalyRatio * float64(m)))
	anomCeil := int(math.Ceil(f.Option.AnomalyRatio * float64(m)))
	f.AnomalyBound = (temp[anomFloor] + temp[anomCeil]) / 2

	if f.AnomalyBound > 0.5 {
		for i := 0; i < m; i++ {
			if f.AnomalyScores[i] > f.AnomalyBound {
				f.Labels[i] = 1
			} else {
				f.Labels[i] = 0
			}
		}
	} // else

	f.Tested = true

	return nil

}

// Predict computes anomaly scores for given dataset and classifies each vector
// as 'normal' or 'anomaly'.
func (f *Forest) Predict(X mat.Matrix) (labels []int, scores []float64, err error) {

	if !f.Trained {
		return nil, nil, errors.New("cannot predict - model has not been trained yet")
	}
	if !f.Tested {
		return nil, nil, errors.New("cannot predict - model has not been tested yet")
	}
	//m represent number of instance
	_, m := X.Dims()
	labels = make([]int, m)
	scores = make([]float64, m)

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
		//s= s - 0.5
		scores[i] = s
	}

	for i := 0; i < m; i++ {
		if scores[i] > f.AnomalyBound {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	return

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
		decoder := json.NewDecoder(file)
		err = decoder.Decode(&f)
	}
	file.Close()

	return err
}
