package tfutil

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

//SaveScatter, save scatter data to png file named outfileName
func SaveScatter(outfileName string, x, y []float64, thetas ...float64) {

	scatterData := make(plotter.XYs, len(x))
	for i := range scatterData {
		scatterData[i].X = x[i]
		scatterData[i].Y = y[i]
	}

	// Create a new plot, set its title and
	// axis labels.
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = outfileName
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	// Draw a grid behind the data
	p.Add(plotter.NewGrid())

	// Make a scatter plotter and set its style.
	s, err := plotter.NewScatter(scatterData)
	if err != nil {
		panic(err)
	}
	s.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	p.Add(s)
	p.Legend.Add("scatter", s)

	parms := len(thetas)

	if parms >= 1 {
		// make a hyA quadratic function
		h := plotter.NewFunction(func(x float64) float64 {
			if parms == 1 {
				return (thetas[0] * x)
			} else {
				return (thetas[0] + thetas[1]*x)
			}
		})
		h.Color = color.RGBA{B: 255, A: 255}
		p.Add(h)
		// Add the plotters to the plot, with a legend
		// entry for each
		p.Legend.Add("hypothesis", h)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, outfileName+".png"); err != nil {
		panic(err)
	}
}

func SaveCostLine(outfileName string, cost []float64) {

	if len(cost) < 1 {
		return
	}

	costData := make(plotter.XYs, len(cost))
	for i := range costData {
		costData[i].X = float64(i)
		costData[i].Y = cost[i]
	}

	// Create a new plot, set its title and
	// axis labels.
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = outfileName
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	// Draw a grid behind the data
	p.Add(plotter.NewGrid())

	// draw cost line
	// Make a line plotter and set its style.
	l, err := plotter.NewLine(costData)
	if err != nil {
		panic(err)
	}
	l.LineStyle.Width = vg.Points(1)
	//l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	l.LineStyle.Color = color.RGBA{B: 255, A: 255}

	p.Add(l)

	// Add the plotters to the plot, with a legend
	// entry for each
	p.Legend.Add("hypothesis", l)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, outfileName+".png"); err != nil {
		panic(err)
	}
}

//residual plots
func SaveResidualPlot(outfileName string, X *mat.Dense, y, theta mat.Vector) {

	var h, diff mat.VecDense
	h.MulVec(X, theta)

	diff.SubVec(y, &h)

	l := y.Len()

	hyXYs, yXYs, diffXYs := make(plotter.XYs, l), make(plotter.XYs, l), make(plotter.XYs, l)
	for i := 0; i < l; i++ {
		hyXYs[i].X = float64(i)
		hyXYs[i].Y = h.AtVec(i)

		yXYs[i].X = float64(i)
		yXYs[i].Y = y.AtVec(i)

		diffXYs[i].X = float64(i)
		diffXYs[i].Y = diff.AtVec(i)
	}

	// Create a new plot, set its title and
	// axis labels.
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = outfileName
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	// Draw a grid behind the data
	p.Add(plotter.NewGrid())

	// draw cost line
	// Make a line plotter and set its style.
	hyL, err := plotter.NewLine(hyXYs)
	if err != nil {
		panic(err)
	}
	hyL.LineStyle.Width = vg.Points(1)
	//l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	hyL.LineStyle.Color = color.RGBA{B: 255, A: 255}

	yL, err := plotter.NewLine(yXYs)
	if err != nil {
		panic(err)
	}
	yL.LineStyle.Width = vg.Points(1)
	//l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	yL.LineStyle.Color = color.RGBA{R: 255, B: 128, A: 255}

	diffL, err := plotter.NewLine(diffXYs)
	if err != nil {
		panic(err)
	}
	diffL.LineStyle.Width = vg.Points(1)
	diffL.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	diffL.LineStyle.Color = color.RGBA{R: 128, B: 128, A: 128}

	p.Add(hyL, yL, diffL)

	// Add the plotters to the plot, with a legend
	// entry for each
	p.Legend.Add("hyL", hyL)
	p.Legend.Add("yL", yL)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, outfileName+".png"); err != nil {
		panic(err)
	}
}

func ShowToImage(x, y []float64,
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
