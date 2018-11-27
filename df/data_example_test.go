package df_test

import (
	"fmt"
	"path"

	. "github.com/flyingyizi/tfutil/df"
	"gonum.org/v1/gonum/mat"
)

func ExampleFeatureScalingMatrix() {

	m := mat.NewDense(5, 3, []float64{

		2104, 3, 399900,
		1600, 3, 329900,
		2400, 3, 369000,
		1416, 2, 232000,
		3000, 4, 539900,
	})

	got := FeatureScalingMatrix(m)
	fk := mat.Formatted(got, mat.Prefix(""), mat.Squeeze())
	fmt.Println(fk)

	//output:
	// ⎡                  0                   0   0.22965393988401123⎤
	// ⎢-0.7924998530535887                   0   -0.3944056793660193⎥
	// ⎢ 0.4654364216346473                   0  -0.04582380632778795⎥
	// ⎢ -1.081825196231883  -1.414213562373095   -1.2671976325742762⎥
	// ⎣ 1.4088886276508243   1.414213562373095    1.4777731783840722⎦

}

// >>> path = 'abc.txt'
// >>> data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
// >>> data2.head()
// >>> data2.head()
//    Size  Bedrooms   Price
// 0  2104         3  399900
// 1  1600         3  329900
// 2  2400         3  369000
// 3  1416         2  232000
// 4  3000         4  539900
// >>> data2 = (data2 - data2.mean()) / data2.std()
// >>> data2.head()
//        Size  Bedrooms     Price
// 0  0.000000  0.000000  0.229654
// 1 -0.792500  0.000000 -0.394406
// 2  0.465436  0.000000 -0.045824
// 3 -1.081825 -1.414214 -1.267198
// 4  1.408889  1.414214  1.477773
// >>>

func ExampleHorizJoinDense() {

	filename := "ex1data2.txt"
	orig := mat.NewDense(CsvToArray(path.Join("testdata", filename)))
	or, oc := orig.Dims()
	// assign Y
	var Y mat.VecDense
	Y.CloneVec(orig.ColView(oc - 1))
	// assign Y
	ones := mat.NewVecDense(or, Ones(or))
	X := HorizJoinDense(ones, orig.Slice(0, or, 0, oc-1)) //X shape is: 'or by (oc)'

	fx := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fyt := mat.Formatted(Y.T(), mat.Prefix(""), mat.Squeeze())
	fmt.Println(fx)
	fmt.Println(fyt)
	//output:
	// ⎡1  2104  3⎤
	// ⎢1  1600  3⎥
	// ⎢1  2400  3⎥
	// ⎢1  1416  2⎥
	// ⎢1  3000  4⎥
	// ⎢1  1985  4⎥
	// ⎢1  1534  3⎥
	// ⎢1  1427  3⎥
	// ⎢1  1380  3⎥
	// ⎢1  1494  3⎥
	// ⎢1  1940  4⎥
	// ⎢1  2000  3⎥
	// ⎢1  1890  3⎥
	// ⎢1  4478  5⎥
	// ⎢1  1268  3⎥
	// ⎢1  2300  4⎥
	// ⎢1  1320  2⎥
	// ⎢1  1236  3⎥
	// ⎢1  2609  4⎥
	// ⎢1  3031  4⎥
	// ⎢1  1767  3⎥
	// ⎢1  1888  2⎥
	// ⎢1  1604  3⎥
	// ⎢1  1962  4⎥
	// ⎢1  3890  3⎥
	// ⎢1  1100  3⎥
	// ⎢1  1458  3⎥
	// ⎢1  2526  3⎥
	// ⎢1  2200  3⎥
	// ⎢1  2637  3⎥
	// ⎢1  1839  2⎥
	// ⎢1  1000  1⎥
	// ⎢1  2040  4⎥
	// ⎢1  3137  3⎥
	// ⎢1  1811  4⎥
	// ⎢1  1437  3⎥
	// ⎢1  1239  3⎥
	// ⎢1  2132  4⎥
	// ⎢1  4215  4⎥
	// ⎢1  2162  4⎥
	// ⎢1  1664  2⎥
	// ⎢1  2238  3⎥
	// ⎢1  2567  4⎥
	// ⎢1  1200  3⎥
	// ⎢1   852  2⎥
	// ⎢1  1852  4⎥
	// ⎣1  1203  3⎦
	//[399900  329900  369000  232000  539900  299900  314900  198999  212000  242500  239999  347000  329999  699900  259900  449900  299900  199900  499998  599000  252900  255000  242900  259900  573900  249900  464500  469000  475000  299900  349900  169900  314900  579900  285900  249900  229900  345000  549000  287000  368500  329900  314000  299000  179900  299900  239500]
}

func ExamplePCA() {

	a := mat.NewDense(2, 5,
		[]float64{-1, -2, -1, 0, 0, 0, 2, 1, 0, 1})
	k := 1
	p := NewPCA(k)
	tra, _ := p.FitTransform(a)
	fmt.Printf("%0.4f", mat.Formatted(tra, mat.Prefix(""), mat.Squeeze()))

	//fmt.Println(p.ExplainedVariance())

	//output:
	// [-0.7071  -2.8284  -1.4142  0.0000  -0.7071]

}
