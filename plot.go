package tfutil

import (
	"errors"
	"fmt"
	"image/color"

	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

type ScaterData struct {
	//using key to contact the xys and colors, and key will be
	// the name of the scatter data
	XYsList map[string]plotter.XYs
	Colors  map[string]color.RGBA
}

func (s *ScaterData) Add(name string, xs, ys []float64) error {
	if len(xs) != len(ys) {
		return (errors.New("wrong length"))
	}
	// init xys data
	xys := make(plotter.XYs, len(xs))
	for i := 0; i < len(xs); i++ {
		xys[i].X, xys[i].Y = xs[i], ys[i]
	}

	if s.XYsList == nil {
		s.XYsList = make(map[string]plotter.XYs)
	}

	if _, ok := s.XYsList[name]; ok {
		delete(s.XYsList, name)
	}
	s.XYsList[name] = xys
	return nil
}

func (s *ScaterData) Del(name string) {
	if s.XYsList == nil {
		return
	}

	if _, ok := s.XYsList[name]; ok {
		delete(s.XYsList, name)
	}
	return
}
func (s *ScaterData) SetColor(name string, c color.RGBA) {
	if s.Colors == nil {
		s.Colors = make(map[string]color.RGBA)
	}

	s.Colors[name] = c
}

func (s *ScaterData) Clear() {

	for k, _ := range s.XYsList {
		delete(s.XYsList, k)
	}
	for k, _ := range s.Colors {
		delete(s.XYsList, k)
	}
	return
}

//SaveScatter, save scatter data to png file named outfileName
func SaveScatter(outfileName string, xys *ScaterData, thetas ...float64) {
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

	for name, sdata := range xys.XYsList {
		// Make a scatter plotter and set its style.
		s, err := plotter.NewScatter(sdata)
		if err != nil {
			panic(err)
		}

		if v, ok := xys.Colors[name]; ok {
			s.GlyphStyle.Color = v
		}
		p.Add(s)
		p.Legend.Add(fmt.Sprint("", name), s)
	}

	parms := len(thetas)
	if parms >= 1 {
		// make a hyA quadratic function
		h := plotter.NewFunction(func(x float64) float64 {
			if parms == 1 {
				return (thetas[0])
			} else if parms == 2 {
				return (thetas[0] + thetas[1]*x)
			} else {
				return -(thetas[0] + thetas[1]*x) / thetas[2]
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

// SaveTwoScatter show two scatter plot 2 by 1
func SaveTwoScatter(outfileName string, xys0, xys1 *ScaterData) {
	const rows, cols = 2, 1
	plots := make([][]*plot.Plot, rows)
	for j := 0; j < rows; j++ {
		plots[j] = make([]*plot.Plot, cols)
		for i := 0; i < cols; i++ {

			p, err := plot.New()
			if err != nil {
				panic(err)
			}
			// draw xys0 scatter data and hypothesis function
			if i == 0 && j == 0 {

				for name, sdata := range xys0.XYsList {
					// Make a scatter plotter and set its style.
					s, err := plotter.NewScatter(sdata)
					if err != nil {
						panic(err)
					}

					if v, ok := xys0.Colors[name]; ok {
						s.GlyphStyle.Color = v
					}
					p.Add(s)
					p.Legend.Add(fmt.Sprint("", name), s)
				}
				//p.Title =
			}
			// draw xys1 scatter data and hypothesis function
			if i == 0 && j == 1 {
				for name, sdata := range xys1.XYsList {
					// Make a scatter plotter and set its style.
					s, err := plotter.NewScatter(sdata)
					if err != nil {
						panic(err)
					}

					if v, ok := xys1.Colors[name]; ok {
						s.GlyphStyle.Color = v
					}
					p.Add(s)
					p.Legend.Add(fmt.Sprint("", name), s)
				}
			}

			// make sure the horizontal scales match
			//p.X.Min = 0
			//			p.X.Max = 5

			plots[j][i] = p
		}
	}

	img := vgimg.New(vg.Points(450), vg.Points(575))
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

	w, err := os.Create(outfileName + ".png")
	if err != nil {
		panic(err)
	}

	png := vgimg.PngCanvas{Canvas: img}
	if _, err := png.WriteTo(w); err != nil {
		panic(err)
	}
}
