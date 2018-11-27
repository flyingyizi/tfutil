package tfutil_test

import (
	"fmt"
	"math/rand"
	"path"
	"time"

	"github.com/flyingyizi/tfutil"
	"github.com/flyingyizi/tfutil/df"

	//	. "github.com/flyingyizi/tfutil/logicregression"

	"gonum.org/v1/gonum/mat"
)

func ExampleSaveScatterToImage_ex3data1() {
	filename := "ex3data1.txt"

	r, c, origx := df.CsvToArray(path.Join("logicregression", "testdata", "X"+filename))
	orig := mat.NewDense(r, c, origx)

	generateRangeNum := func(min, max int) int {
		rand.Seed(time.Now().Unix())
		randNum := rand.Intn(max-min) + min
		return randNum
	}

	rnd := generateRangeNum(0, (r - 1))
	data := orig.RawRowView(rnd)

	tfutil.SaveScatterToImage(fmt.Sprintf("%s-image-%d", filename, rnd), 20, 20, data)
	//output:
	//
}
