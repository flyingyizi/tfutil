package tfutil

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"

	"gonum.org/v1/gonum/floats"

	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

//ScatterData store scatter data
type ScatterData struct {
	//using key to contact the xys and colors, and key will be
	// the name of the scatter data
	XYZsList map[string]plotter.XYZs
}

//Add add new item to scatter data list, the new item marked by name
func (s *ScatterData) Add(name string, xs, ys, zs []float64) error {
	if len(xs) != len(ys) || len(xs) != len(zs) {
		return (errors.New("wrong length"))
	}
	// init xyzs data
	xyzs := make(plotter.XYZs, len(xs))
	for i := 0; i < len(xs); i++ {
		xyzs[i].X, xyzs[i].Y, xyzs[i].Z = xs[i], ys[i], zs[i]
	}

	if s.XYZsList == nil {
		s.XYZsList = make(map[string]plotter.XYZs)
	}

	if _, ok := s.XYZsList[name]; ok {
		delete(s.XYZsList, name)
	}
	s.XYZsList[name] = xyzs
	return nil
}

//Del delete item marked by name from the catter data list
func (s *ScatterData) Del(name string) {
	if s.XYZsList == nil {
		return
	}

	if _, ok := s.XYZsList[name]; ok {
		delete(s.XYZsList, name)
	}
	return
}

//Clear clear scatter data list
func (s *ScatterData) Clear() {

	for k := range s.XYZsList {
		delete(s.XYZsList, k)
	}
	return
}

//SaveScatter  save scatter data to png file named outfileName
func SaveScatter(outfileName string, xys *ScatterData, thetas ...float64) {
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

	for name, sdata := range xys.XYZsList {
		// Make a scatter plotter and set its style.
		s, err := plotter.NewScatter(sdata)
		if err != nil {
			panic(err)
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

//SaveLine  save line ,the line's X is index of slice, the Y is item's value
//for example , <i, line[i]> is a pixel in the line
func SaveLine(outfileName string, line []float64) {

	if len(line) < 1 {
		return
	}

	costData := make(plotter.XYZs, len(line))
	for i := range costData {
		costData[i].X = float64(i)
		costData[i].Y = line[i]
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
	//p.Legend.Add("hypothesis", l)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, outfileName+".png"); err != nil {
		panic(err)
	}
}

//SaveResidualPlot save residual plots to png file named outfileName
func SaveResidualPlot(outfileName string, X *mat.Dense, y, theta mat.Vector) {

	var h, diff mat.VecDense
	h.MulVec(X, theta)

	diff.SubVec(y, &h)

	l := y.Len()

	hyXYs, yXYs, diffXYs := make(plotter.XYZs, l), make(plotter.XYZs, l), make(plotter.XYZs, l)
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

// SaveScatters show multi scatters
// if samePlot is true, all of scatters will be show in one plot,otherwise will be
// show in separate plot
func SaveScatters(outfileName string, scatters ...*ScatterData) {

	rows, cols := func() (int, int) {
		t := (len(scatters) / 2)
		if t == 0 {
			return 1, 1
		} else if len(scatters) > t*2 {
			return t + 1, 2
		}
		return t, 2
	}()

	plots := make([][]*plot.Plot, rows)
	for j := 0; j < rows; j++ {
		plots[j] = make([]*plot.Plot, cols)
	}

	for i, xyzs := range scatters {
		p, err := plot.New()
		if err != nil {
			panic(err)
		}
		for name, sdata := range xyzs.XYZsList {
			// Make a scatter plotter and set its style.
			s, err := myNewScatter(sdata)
			if err != nil {
				panic(err)
			}
			p.Add(s)
			p.Legend.Add(fmt.Sprint("", name), s)
		}
		//i:=k*row + col
		row, col := i/2, i%2
		plots[row][col] = p
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

func myNewScatter(sdata plotter.XYZs) (s *plotter.Scatter, err error) {
	// Make a scatter plotter and set its style.
	s, err = plotter.NewScatter(sdata)
	if err != nil {
		return
	}

	// Calculate the range of Z values.
	minZ, maxZ := math.Inf(1), math.Inf(-1)
	for _, xyz := range sdata {
		if xyz.Z > maxZ {
			maxZ = xyz.Z
		}
		if xyz.Z < minZ {
			minZ = xyz.Z
		}
	}

	//
	colors := moreland.Kindlmann() // Initialize a color map.
	colors.SetMax(1 + maxZ)
	colors.SetMin(minZ)

	// Specify style and color for individual points.
	s.GlyphStyleFunc = func(i int) draw.GlyphStyle {
		_, _, z := sdata.XYZ(i)
		d := (z - minZ) / (maxZ - minZ)
		rng := maxZ - minZ
		k := d*rng + minZ
		c, err := colors.At(k)
		if err != nil {
			panic(err)
		}
		return draw.GlyphStyle{Color: c, Radius: vg.Points(3), Shape: draw.CircleGlyph{}}
	}

	return
}

// func myNewLegend(p *plot.Plot, minZ,maxZ float64) {
// //////////
// 	//Create a legend
// 	thumbs := plotter.PaletteThumbnailers(colors.Palette(n))
// 	for i := len(thumbs) - 1; i >= 0; i-- {
// 		t := thumbs[i]
// 		if i != 0 && i != len(thumbs)-1 {
// 			p.Legend.Add("", t)
// 			continue
// 		}
// 		var val int
// 		switch i {
// 		case 0:
// 			val = int(minZ)
// 		case len(thumbs) - 1:
// 			val = int(maxZ)
// 		}
// 		p.Legend.Add(fmt.Sprintf("%d", val), t)
// 	}

// 	// This is the width of the legend, experimentally determined.
// 	const legendWidth = vg.Centimeter
// 	// Slide the legend over so it doesn't overlap the ScatterPlot.
// 	p.Legend.XOffs = legendWidth

// //////////

// }

//SaveScatterToImage deal the data as a image matrix,
//it will draw all of item in the data as a pixel
//each item in the data will be dealed as the Gray intensity
func SaveScatterToImage(outfileName string, r, c int, data []float64) (err error) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	//p.Title.Text = "A Logo"

	img, err := NewGray(r, c, data)
	if err != nil {
		return
	}

	p.Add(img)

	err = p.Save(5*vg.Centimeter, 5*vg.Centimeter, outfileName+".png")

	return
}

// NewGray create a plotter image from scatter data
func NewGray(r, c int, data []float64) (img *plotter.Image, err error) {
	if r*c != len(data) {
		err = errors.New("wrong scatter size")
		return
	}

	pic := image.NewGray(image.Rect(0, 0, r, c))
	// assign backgroud to white
	for x := 0; x < r; x++ {
		for y := 0; y < c; y++ {
			pic.SetGray(x, y, color.Gray{255})
		}
	}

	min, max := floats.Min(data), floats.Max(data)
	grayScope := max - min
	for x := 0; x < r; x++ {
		for y := 0; y < c; y++ {
			// scaling data to 0~255
			d := 255 - math.Round(255*(data[x*c+y]-min)/grayScope)
			pic.SetGray(x, y, color.Gray{uint8(d)})
		}
	}
	img = plotter.NewImage(pic, 0, 0, float64(r), float64(c))
	return
}
