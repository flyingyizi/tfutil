package csvdata

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"

	"gonum.org/v1/gonum/mat"
)

// CsvToArray read csv to matrix with shape r by c
func CsvToArray(filename string) (r, c int, out []float64) {
	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)

	reader.Comment = '#' //可以设置读入文件中的注释符
	reader.Comma = ','   //默认是逗号，也可以自己设置

	data := make([]string, 0)
	r, c = 0, 0
	if firstRecord, err := reader.Read(); err == io.EOF {
		return
	} else if err != nil {
		fmt.Println("Error:", err)
		return
	} else {
		c = len(firstRecord)
		data = append(data, firstRecord...)
	}

	// continue scan
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println("Error:", err)
			return
		}
		data = append(data, record...)
	}

	total, out := len(data), make([]float64, len(data))
	for i := 0; i < total; i++ {
		if value, err := strconv.ParseFloat(data[i], 64); err == nil {
			out[i] = value
		} else {
			return
		}
	}

	if total%c != 0 {
		return 0, 0, nil
	}
	r = total / c

	return

}

//FeatureScalingMatrix normalize each colum in the matrix
//with algothim $\frac {x-m} {stderr}$
func FeatureScalingMatrix(m mat.Matrix) *mat.Dense {
	if m == nil {
		return nil
	}

	var t mat.Dense
	t.Clone(m.T())

	tr, _ := t.Dims()
	for i := 0; i < tr; i++ {
		t.SetRow(i, FeatureScaling(t.RawRowView(i)...))
	}
	var o mat.Dense
	o.Clone(t.T())

	return &o
}

//FeatureScaling feature scaling  with algothim $\frac {x-m} {stderr}$
// 特征归一化(Feature Scaling)
func FeatureScaling(data ...float64) []float64 {
	if len(data) == 0 {
		return nil
	}
	out := make([]float64, len(data))
	copy(out, data)

	m := stat.Mean(data, nil)
	s := stat.StdDev(data, nil)

	floats.AddConst(-1*m, out)
	//	if !math.IsNaN(s) && s != 0.0 {
	if s != 0.0 {
		floats.Scale(1/s, out)
	}

	return out
}

//HorizJoinDense join a and b with same row
//
func HorizJoinDense(a mat.Matrix, bs ...mat.Matrix) (dest *mat.Dense) {
	ar, ac := a.Dims()

	for _, j := range bs {
		jr, _ := j.Dims()
		if ar != jr {
			panic("wrong row size")
		}
	}

	var x mat.Dense
	x.Clone(a)

	//help function
	getColOfb := func(b mat.Matrix, j int) []float64 {
		br, _ := b.Dims()
		out := make([]float64, br)
		for i := 0; i < br; i++ {
			out[i] = b.At(i, j)
		}
		return out
	}

	var (
		poi *mat.Dense
		ok  bool
	)

	for _, bitem := range bs {
		_, bc := bitem.Dims()

		poi, ok = x.Grow(0, bc).(*mat.Dense)
		if ok != true {
			panic("type of a is wrong")
		}
		//append all coloums from bitem
		for j := 0; j < bc; j++ {
			poi.SetCol(ac, getColOfb(bitem, j))
			ac = ac + 1
		}
		x = *poi
	}

	return poi
}

//VerticalJoinDense join a and b with same colum
//
func VerticalJoinDense(a mat.Matrix, bs ...mat.Matrix) (dest *mat.Dense) {
	ar, ac := a.Dims()

	for _, j := range bs {
		_, jc := j.Dims()
		if ac != jc {
			panic("wrong colum size")
		}
	}

	//help function
	getRowOfb := func(b mat.Matrix, j int) []float64 {
		_, bc := b.Dims()
		out := make([]float64, bc)
		for i := 0; i < bc; i++ {
			out[i] = b.At(j, i)
		}
		return out
	}

	var x mat.Dense
	x.Clone(a)

	var (
		poi *mat.Dense
		ok  bool
	)

	for _, bitem := range bs {
		br, _ := bitem.Dims()

		poi, ok = x.Grow(br, 0).(*mat.Dense)
		if ok != true {
			panic("type of a is wrong")
		}
		//append all coloums from bitem
		for j := 0; j < br; j++ {
			poi.SetRow(ar, getRowOfb(bitem, j))
			ar = ar + 1
		}
		x = *poi
	}

	return poi
}

//Flatten flatten  two-dimensional array to one dimensional
func Flatten(f [][]float64) (r, c int, d []float64) {
	r = len(f)
	if r == 0 {
		panic("bad test: no row")
	}
	c = len(f[0])
	d = make([]float64, 0, r*c)
	for _, row := range f {
		if len(row) != c {
			panic("bad test: ragged input")
		}
		d = append(d, row...)
	}
	return r, c, d
}

//Unflatten  unflatten one dimensional to two-dimensional array
// with shape r by c
func Unflatten(r, c int, d []float64) [][]float64 {
	m := make([][]float64, r)
	for i := 0; i < r; i++ {
		m[i] = d[i*c : (i+1)*c]
	}
	return m
}

// EncodeOneHot covert origLabels to onehot
// each yoneHot's coloum map to one label's onehot in origLabels
func EncodeOneHot(labelNum int, origLabels []float64) (yOnehot *mat.Dense) {

	rmDuplicate := func(list *[]float64) []float64 {
		var x []float64 = []float64{}
		for _, i := range *list {
			if len(x) == 0 {
				x = append(x, i)
			} else {
				for k, v := range x {
					if i == v {
						break
					}
					if k == len(x)-1 {
						x = append(x, i)
					}
				}
			}
		}
		return x
	}

	labels := rmDuplicate(&origLabels)
	maps := make(map[float64]int)
	for i, j := range labels {
		maps[j] = i
	}

	if len(labels) > labelNum {
		panic("labelNum not enough big to cover origLabels")
	}

	eye := Eye(labelNum)

	yOnehot = mat.NewDense(labelNum, len(origLabels), nil)

	for i := 0; i < len(origLabels); i++ {
		key := origLabels[i]
		index := maps[key]
		yOnehot.SetCol(i, eye.RawRowView(index))
	}
	return
}

// Eye returns a new identity matrix of size n×n.
func Eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return mat.NewDense(n, n, d)
}

//Ones generate  one dimensional array with n length
// the array filled with one
func Ones(n int) []float64 {
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = 1
	}
	return d
}

// //还可以设置以下信息
// //FieldsPerRecord  int  // Number of expected fields per record
// //LazyQuotes       bool // Allow lazy quotes
// //TrailingComma    bool // Allow trailing comma
// //TrimLeadingSpace bool // Trim leading space
// //line             int
// //column           int

// fout, err := os.OpenFile("out.txt", os.O_RDWR|os.O_APPEND|os.O_CREATE, 0666)
// if err != nil {
// 	fmt.Println("Error:", err)
// 	return
// }
// defer fout.Close()
// content := ""
// k := 0 //第一行是字段名，不需要

// for {
// 	record, err := reader.Read()
// 	if err == io.EOF {
// 		break
// 	} else if err != nil {
// 		fmt.Println("Error:", err)
// 		return
// 	}

// 	if k > 0 { //record是[]strings， 怎样直接获得域值
// 		for _, v := range record {
// 			tmp := strings.Split(v, "|")
// 			//fmt.Print("<" + tmp[1] + ">" + tmp[7] + "</" + tmp[1] + ">")
// 			content = content + "<" + tmp[1] + ">" + tmp[7] + "</" + tmp[1] + ">"
// 		}
// 	}
// 	k = k + 1
// }
// fmt.Printf("\n")
// fout.WriteString(content + "\n")
// fmt.Printf("\n")
