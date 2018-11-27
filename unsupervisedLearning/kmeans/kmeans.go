package kmeans

import (
	"math"
	"math/rand"

	"github.com/flyingyizi/tfutil/df"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

//Kmeans return
//
// num of samples is row of X
// k is cluserNum
// output: idx store each sample's cluster index based on centroids
func Kmeans(X *mat.Dense, k int, initCentroids *mat.Dense, numEpochs int) (idx []int) {
	if initCentroids == nil {
		initCentroids = Seed(X, k)
	}

	_, xc := X.Dims()
	_, initc := initCentroids.Dims()
	cr, _ := initCentroids.Dims()
	if k != cr || xc != initc {
		panic("wrong size")
	}

	var centroids mat.Dense
	centroids.Clone(initCentroids)
	p := &centroids

	for i := 0; i < numEpochs; i++ {
		idx = findClosestCentroids(X, k, p)
		//move centroid
		p = computeCentroids(X, k, idx)
	}

	return
}

// Seed initializing randomly the seeds
//
//
func Seed(X *mat.Dense, k int) (initCentroids *mat.Dense) {

	xr, xc := X.Dims()
	initCentroids = mat.NewDense(k, xc, nil)

	for i := 0; i < k; i++ {
		ra := rand.Intn(xr)
		initCentroids.SetRow(i, X.RawRowView(ra))
	}
	return
}

//findClosestCentroids is
// num of samples is row of X
// k is cluserNum
// row index of centroids dealed as the cluster's index
// output: idx store each sample's cluster index based on centroids
func findClosestCentroids(X *mat.Dense, k int, centroids *mat.Dense) (idx []int) {

	m, xc := X.Dims()
	cr, cc := centroids.Dims()
	if xc != cc || cr != k {
		panic("wrong input size")
	}
	t := make([]float64, xc)
	idx = make([]int, m)

	for i := 0; i < m; i++ {
		minDist := math.Inf(1)
		for j := 0; j < k; j++ {
			// use L2 norm as the distance
			floats.SubTo(t, X.RawRowView(i), centroids.RawRowView(j))
			dist := floats.Norm(t, 2)

			if dist < minDist {
				minDist = dist
				idx[i] = j
			}
		}
	}
	return
}

//computeCentroids  compute and return new centroids
// num of samples is row of X
func computeCentroids(X *mat.Dense, k int, idx []int) (centroids *mat.Dense) {
	m, xc := X.Dims()

	if m != len(idx) {
		panic("wrong size")
	}

	sum := make([][]float64, k)
	// store accumulate sample that with same clusterNo
	accuCluster := make([]int, k)
	for i := range sum {
		sum[i] = make([]float64, xc)
	}

	// accumulate each cluater
	for j := 0; j < len(idx); j++ {
		clusterNo := idx[j]
		floats.Add(sum[clusterNo], X.RawRowView(j))
		accuCluster[clusterNo] = accuCluster[clusterNo] + 1
	}
	// sum means
	for i := 0; i < k; i++ {
		no := accuCluster[i]

		if no != 0 {
			floats.Scale(1.0/float64(no), sum[i])
		}
	}

	centroids = mat.NewDense(df.Flatten(sum))
	return
}
