package df

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// PCA calculates the principal components of a matrix, or the axis of greatest variance and
// then projects matrices onto those axis.
// See https://en.wikipedia.org/wiki/Principal_component_analysis for further details.
type PCA struct {
	// K is the number of components
	K  int
	pc *stat.PC
}

// NewPCA constructs a new Principal Component Analysis transformer to reduce the dimensionality,
// projecting matrices onto the axis of greatest variance
func NewPCA(k int) *PCA {
	return &PCA{K: k, pc: &stat.PC{}}
}

// Fit calculates the principal component directions (axis of greatest variance) within the
// training data which can then be used to project matrices onto those principal components using
// the Transform() method.
func (p *PCA) Fit(m mat.Matrix) *PCA {
	if ok := p.pc.PrincipalComponents(m.T(), nil); !ok {
		panic("nlp: PCA analysis failed during fitting")
	}

	return p
}

// Transform projects the matrix onto the first K principal components calculated during training
// (the Fit() method).  The returned matrix will be of reduced dimensionality compared to the input
// (K x c compared to r x c of the input).
func (p *PCA) Transform(m mat.Matrix) (mat.Matrix, error) {
	r, _ := m.Dims()

	//var proj mat.Dense
	var proj mat.Dense
	proj.Mul(m.T(), p.pc.VectorsTo(nil).Slice(0, r, 0, p.K))
	return proj.T(), nil
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
func (p *PCA) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	return p.Fit(m).Transform(m)
}

// ExplainedVariance returns a slice of float64 values representing the variances of the
// principal component scores.
func (p *PCA) ExplainedVariance() []float64 {
	return p.pc.VarsTo(nil)
}
