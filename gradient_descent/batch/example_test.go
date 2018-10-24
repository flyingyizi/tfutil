package batch

import (
	"image/color"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func ExampleGradientDescent_ex1data1() {

	Xd, Yd := loadData("ex1data1.txt")

	data := []float64{} // 为后面向量化计算，会在其中增加 `x^0` 即等于1的一列数据
	x1 := []float64{}   // 存放图形显示原始数据
	for _, j := range Xd {
		data = append(data, 1)
		data = append(data, j...)

		x1 = append(x1, j...)
	}
	r, c := len(Xd), len(Xd[0])+1
	X1 := mat.NewDense(r, c, data)
	y1 := mat.NewVecDense(r, Yd)
	theta1 := mat.NewVecDense(c, nil)

	t, cost := GradientDescent(X1, y1, theta1, 0.01, 1000)

	showToImage(x1, Yd, cost, "ex1data1_gradientDescent", t...)

	// Output:
}

func ExampleGradientDescent_ex1data2() {

	Xd, Yd := loadData("ex1data2.txt")

	data := []float64{} // 为后面向量化计算，会在其中增加 `x^0` 即等于1的一列数据
	x1 := []float64{}   // 存放图形显示原始数据
	for _, j := range Xd {
		data = append(data, 1)
		data = append(data, j...)

		x1 = append(x1, j...)
	}
	r, c := len(Xd), len(Xd[0])+1
	X1 := mat.NewDense(r, c, data)
	y1 := mat.NewVecDense(r, Yd)
	theta1 := mat.NewVecDense(c, nil)

	t, cost := GradientDescent(X1, y1, theta1, 0.01, 1000)

	showToImage(x1, Yd, cost, "ex1data2_gradientDescent", t...)

	// Output:
}

func showToImage(x, y []float64,
	cost []float64,
	filename string, thetas ...float64) {

	showScatter := true
	//prepare scatter
	if len(x) != len(y) {
		showScatter = false
		//panic(errors.New("wrong shape size"))
	}

	var scatterData plotter.XYs
	if showScatter == true {
		scatterData := make(plotter.XYs, len(x))
		for i := range scatterData {
			scatterData[i].X = x[i]
			scatterData[i].Y = y[i]
		}
	}
	costData := make(plotter.XYs, len(cost))
	for i := range costData {
		costData[i].X = float64(i)
		costData[i].Y = cost[i]
	}

	const rows, cols = 2, 1
	plots := make([][]*plot.Plot, rows)
	for j := 0; j < rows; j++ {
		plots[j] = make([]*plot.Plot, cols)
		for i := 0; i < cols; i++ {

			p, err := plot.New()
			if err != nil {
				panic(err)
			}

			// draw scatter data and hypothesis function
			if i == 0 && j == 0 {
				// Make a scatter plotter and set its style.
				s, err := plotter.NewScatter(scatterData)
				if err != nil {
					panic(err)
				}
				s.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}

				// make a hyA quadratic function x^2
				h := plotter.NewFunction(func(x float64) float64 {
					var total float64 = 0.0
					for index, t := range thetas {
						total += t * math.Pow(x, float64(index))
					}
					return total
				})
				h.Color = color.RGBA{B: 255, A: 255}

				p.Add(s, h)
			}
			// draw cost line
			if i == 0 && j == 1 {
				// Make a line plotter and set its style.
				l, err := plotter.NewLine(costData)
				if err != nil {
					panic(err)
				}
				l.LineStyle.Width = vg.Points(1)
				//l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
				l.LineStyle.Color = color.RGBA{B: 255, A: 255}

				p.Add(l)
			}

			// make sure the horizontal scales match
			//p.X.Min = 0
			//			p.X.Max = 5

			plots[j][i] = p
		}
	}

	img := vgimg.New(vg.Points(150), vg.Points(175))
	dc := draw.New(img)

	t := draw.Tiles{
		Rows: rows,
		Cols: cols,
	}

	canvases := plot.Align(plots, t, dc)
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			if plots[j][i] != nil {
				plots[j][i].Draw(canvases[j][i])
			}
		}
	}

	w, err := os.Create(filename + ".png")
	if err != nil {
		panic(err)
	}

	png := vgimg.PngCanvas{Canvas: img}
	if _, err := png.WriteTo(w); err != nil {
		panic(err)
	}
}
