package batch

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

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
