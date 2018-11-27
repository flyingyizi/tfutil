package iforest

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Tree is base structure for the iTree
type Tree struct {
	Root *Node
}

// Node is a structure for iNode
type Node struct {
	Left      *Node
	Right     *Node
	Split     float64
	Attribute int
	C         float64
	Size      int
	External  bool
}

// BuildTree builds decision tree for given data, attributes and values used for
// splits are chosen randomly
// the given data specialized by<X, ind>
func (t *Tree) BuildTree(X mat.Matrix, ind []int, maxDepth int) error {
	xr, _ := X.Dims()

	//radom select a feature in all features
	att := rand.Intn(xr)
	//radom select a value of the feature specialized by att in the
	//sub-sample identified by <X,ind>
	split := radomChooseSplit(X, ind, att)

	root := &Node{Split: split, Attribute: att, Size: len(ind)}

	l, r := splitSample(X, ind, split, att)

	root.External = false
	root.Left = newNode(X, l, maxDepth, 1)
	root.Right = newNode(X, r, maxDepth, 1)

	t.Root = root

	return nil
}

// PathLength computes the length of the path for given data vector.
// The result is number of edges from the root to terminating node
// plus adjustment value - c(Size) as described in algorithm specification.
func (t *Tree) PathLength(V []float64) (pathLength float64) {

	var currentNode, nextNode *Node
	currentNode = t.Root

	pathLength = 0
	for {
		pathLength++
		if V[currentNode.Attribute] < currentNode.Split {
			nextNode = currentNode.Left
		} else {
			nextNode = currentNode.Right
		}
		currentNode = nextNode

		if currentNode.Size <= 1 {

			return float64(pathLength)
		}
		if currentNode.External {

			return float64(pathLength) + currentNode.C
		}

	}

}

// newNode create new node in the tree.  It finishes when no more sample is available
// or when the tree reaches its maximum depth.
//
//input: <X, ind> together define the sub-sample for current node,the X's row represent feature, the X's
//       colum represent each instance. the ind slice store the instance index in the sub-sample
//input: maxDepth is the tree's max depth, the node belongs to the tree
func newNode(X mat.Matrix, ind []int, maxDepth int, d int) *Node {
	xr, _ := X.Dims()

	// c(n) = 2H(n − 1) − (2(n − 1)/n), refer to the page 3 in icdm08b.pdf
	var c float64
	if len(ind) > 1 {
		c = computeC(float64(len(ind)))
	}

	//is external node
	if len(ind) <= 1 || d >= maxDepth {
		return &Node{External: true, Size: len(ind), C: c}
	}

	//radom select a feature in all features
	att := rand.Intn(xr)
	split := radomChooseSplit(X, ind, att)

	internalNode := &Node{Attribute: att, Split: split, External: false, Size: len(ind), C: c}

	l, r := splitSample(X, ind, split, att)

	internalNode.Left = newNode(X, l, maxDepth, d+1)

	internalNode.Right = newNode(X, r, maxDepth, d+1)

	return internalNode

}

// splitSample  divides sub-sample on two parts based on chosen att and split value
//
//input: <X, ind> together define the sub-sample,the X's row represent feature, the X's
//       colum represent each instance. the ind slice store the instance index in the sub-sample
//input: att identify which feature will be based to split, split identify the split value
//output: l slice store the instace index that the special feature value less than the split
//output: r slice store the instace index that the special feature value great or equal to the split
func splitSample(X mat.Matrix, ind []int, split float64, att int) (l []int, r []int) {

	l, r = make([]int, 0), make([]int, 0)

	for _, col := range ind {
		v := X.At(att, col)
		if v < split {
			l = append(l, col)
		} else {
			r = append(r, col)
		}
	}

	return l, r
}

// radomChooseSplit randomly choose value of the split. This value is always between
// lowest and highest value among the values specialized by the att(feature).
func radomChooseSplit(X mat.Matrix, ind []int, att int) float64 {
	max, min := -math.MaxFloat64, math.MaxFloat64

	for _, col := range ind {
		val := X.At(att, col)
		if val <= min {
			min = val
		}
		if val >= max {
			max = val
		}
	}
	return min + (max-min)*rand.Float64()

}
