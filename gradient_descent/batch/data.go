package batch

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
// 注意csv文件中的列数据在输出时以行存储，因此在矩阵运算时需要T()后再参与运算
func CsvToDense(filename string) (X *mat.Dense, orign [][]float64) {
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
	orign = make([][]float64, cols+1)
	for i := 0; i < cols+1; i++ {
		orign[i] = make([]float64, row)
		copy(orign[i], data[i])
	}

	for i := 0; i < cols+1; i++ {
		m := stat.Mean(data[i], nil)
		s := stat.StdDev(data[i], nil)
		floats.AddConst(-1*m, data[i])
		floats.Scale(1/s, data[i])
	}

	X = mat.NewDense(cols+1, row, nil)
	for i := 0; i < cols+1; i++ {
		X.SetRow(i, data[i])
	}

	return

}

func loadData(filename string) (outx [][]float64, outy []float64) {

	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)

	reader.Comment = '#' //可以设置读入文件中的注释符
	reader.Comma = ','   //默认是逗号，也可以自己设置

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println("Error:", err)
			return
		}

		vals := []float64{}
		for _, j := range record {
			if value, err := strconv.ParseFloat(j, 64); err == nil {
				vals = append(vals, value)
			} else {
				return
			}
		}
		len := len(vals)
		if len < 2 {
			return
		}
		outx = append(outx, vals[:len-1])
		outy = append(outy, vals[len-1])
	}
	return

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
}
