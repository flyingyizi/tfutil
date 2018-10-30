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

// CsvToDense,
// csv文件中的列数据在输出到orig时,orgin[i]代表原始数据的第i列
// 当normalize为true时，代表输出的X，Y经过正规化处理; 否则数据为向量化后的原始数据
func CsvToDense(filename string, normalize bool) (X *mat.Dense, Y *mat.VecDense, orign [][]float64) {
	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)

	reader.Comment = '#' //可以设置读入文件中的注释符
	reader.Comma = ','   //默认是逗号，也可以自己设置

	firstRecord, err := reader.Read()
	if err == io.EOF {
		return
	} else if err != nil {
		fmt.Println("Error:", err)
		return
	}
	cols := len(firstRecord)
	data := make([][]float64, cols+1)

	row := 0
	for i := 0; i < cols; i++ {
		data[i+1] = make([]float64, 0)
		if value, err := strconv.ParseFloat(firstRecord[i], 64); err == nil {
			data[i+1] = append(data[i+1], value)
		} else {
			return
		}
	}
	row++

	for {
		// continue scan
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println("Error:", err)
			return
		}
		for i, j := range record {
			if value, err := strconv.ParseFloat(j, 64); err == nil {
				data[i+1] = append(data[i+1], value)
			} else {
				return
			}
		}
		row++
	}

	data[0] = make([]float64, row)
	for i := 0; i < row; i++ {
		data[0][i] = 1
	}

	//assign orig data
	orign = make([][]float64, cols)
	for i := 0; i < cols; i++ {
		orign[i] = make([]float64, row)
		copy(orign[i], data[i+1])
	}

	if normalize {
		// ones line dont need feature normalize
		for i := 1; i < cols+1; i++ {
			m := stat.Mean(data[i], nil)
			s := stat.StdDev(data[i], nil)
			floats.AddConst(-1*m, data[i])
			if s != 0.0 {
				floats.Scale(1/s, data[i])
			}
		}
		X = mat.NewDense(row, cols, nil)
		for i := 0; i < cols; i++ {
			X.SetCol(i, data[i])
		}
	} else {
		X = mat.NewDense(row, cols, nil)
		for i := 0; i < cols; i++ {
			X.SetCol(i, data[i])
		}
	}

	Y = mat.NewVecDense(row, data[cols])

	return

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
