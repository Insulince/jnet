package main

// Padding note:
// L: 31 x 51, W: 14, S: 11
// W and S can be flipped for another interesting padding configuration

import (
	"fmt"
	"github.com/Insulince/jnet"
	"math"
	"os"
)

const (
	filterStride  = 1
	poolingStride = 2
)

func main() {
	rows := 31
	cols := 51
	data := [][]float64{}
	for i := 0; i < rows; i++ {
		data = append(data, []float64{})
		for j := 0; j < cols; j++ {
			data[i] = append(data[i], 1.0)
		}
	}

	printData(data)
	fmt.Println()
	p := maxPool(data, 14, 11)
	printData(p)

	os.Exit(0)

	filters := [][][]float64{
		{
			{1, -1, -1},
			{-1, 1, -1},
			{-1, -1, 1},
		},
		{
			{-1, -1, 1},
			{-1, 1, -1},
			{1, -1, -1},
		},
		{
			{1, -1, 1},
			{-1, 1, -1},
			{1, -1, 1},
		},
	}

	input := [][]float64{
		{-1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, 1, -1, -1, -1, -1, -1, 1, -1},
		{-1, -1, 1, -1, -1, -1, 1, -1, -1},
		{-1, -1, -1, 1, -1, 1, -1, -1, -1},
		{-1, -1, -1, -1, 1, -1, -1, -1, -1},
		{-1, -1, -1, 1, -1, 1, -1, -1, -1},
		{-1, -1, 1, -1, -1, -1, 1, -1, -1},
		{-1, 1, -1, -1, -1, -1, -1, 1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1},
	}

	fmt.Println("INPUT:")
	printData(input)
	fmt.Println()

	convolvedOutput := [][][]float64{}
	reluedOutput := [][][]float64{}
	pooledOutput := [][][]float64{}

	for fi := range filters {
		convolvedOutput = append(convolvedOutput, [][]float64{})
		reluedOutput = append(reluedOutput, [][]float64{})
		pooledOutput = append(pooledOutput, [][]float64{})

		quantityConvolves := len(input) - len(filters[fi]) + 1
		for ri := 0; ri < quantityConvolves; ri++ {
			convolvedOutput[fi] = append(convolvedOutput[fi], []float64{})
			reluedOutput[fi] = append(reluedOutput[fi], []float64{})

			for ci := 0; ci < quantityConvolves; ci++ {
				window := [][]float64{}
				for wri := 0; wri < len(filters[fi]); wri++ {
					window = append(window, []float64{})
					for wci := 0; wci < len(filters[fi]); wci++ {
						window[wri] = append(window[wri], input[ri+wri][ci+wci])
					}
				}
				convolvedOutput[fi][ri] = append(convolvedOutput[fi][ri], convolve(window, filters[fi], filterStride))
			}
			for coi := range convolvedOutput[fi][ri] {
				reluedOutput[fi][ri] = append(reluedOutput[fi][ri], relu(convolvedOutput[fi][ri][coi]))
			}
		}

		pooledOutput[fi] = maxPool(reluedOutput[fi], 2, poolingStride)

		fmt.Printf("FILTER #%v:\n", fi+1)
		printData(filters[fi])

		fmt.Println("CONVOLVE:")
		printData(convolvedOutput[fi])

		fmt.Println("+ ReLU:")
		printData(reluedOutput[fi])

		fmt.Println("+ POOL:")
		printData(pooledOutput[fi])

		fmt.Println()
	}
}

func convolve(window [][]float64, filter [][]float64, stride int) float64 {
	var val float64 = 0
	for ri := 0; ri < len(window); ri += stride {
		for ci := 0; ci < len(window); ci++ {
			val += window[ri][ci] * filter[ri][ci]
		}
	}
	return val / float64(len(filter)*len(filter))
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func printData(data [][]float64) {
	output := ""

	for ri, row := range data {
		output += fmt.Sprintf("%3d | ", ri)
		for _, cell := range row {
			output += fmt.Sprintf("%2.f ", cell)
		}
		output += "\n"
	}

	fmt.Print(output)
}

func maxPool(data [][]float64, windowSize int, stride int) [][]float64 {
	// Deep copy the data array so we don't accidentally propagate padding changes back up to the caller.
	data = jnet.Duplicate2d(data)

	dl := len(data[0]) // Length of the data.
	dh := len(data)    // Height of the data.

	// Calculate horizontal and vertical padding needed to accommodate provided window size at provided stride.
	hpad := calculatePadding(dl, windowSize, stride)
	vpad := calculatePadding(dh, windowSize, stride)

	// Split the horizontal and vertical padding evenly on left and right sides and on the top and bottom sides, respectively.
	lpad, rpad := splitPadding(hpad)
	tpad, bpad := splitPadding(vpad)

	for ri := 0; ri < dh; ri++ { // For each row in the data...
		data[ri] = leftPad(data[ri], lpad)
		data[ri] = rightPad(data[ri], rpad)
	}

	dl = len(data[0]) // Recalculate data length since it has been updated with horizontal padding.

	data = topPad(data, tpad)
	data = bottomPad(data, bpad)

	dh = len(data) // Recalculate data height since it has been updated with vertical padding.

	// At this point, data should contain the provided data perfectly centered between enough padding on all sides to accommodate the provided window size at the provided stride.
	printData(data)
	fmt.Println()

	// Perform the pooling operation.
	// TODO: Check the <= nonsense. Output SHOULD be a 3x3 matrix, but without <= its 2x2 for some reason.
	pooledData := [][]float64{}
	for wvsi := 0; wvsi+windowSize <= dh; wvsi += stride { // For every start of a window vertically...
		pooledData = append(pooledData, []float64{})           // Append a new row to be filled with pooling data.
		for whsi := 0; whsi+windowSize <= dl; whsi += stride { // For every start of a window horizontally...
			maxValueInWindow := -math.MaxFloat64    // Search for the maximum value in the window. Start with the smallest value possible so it will always be beat.
			for wri := 0; wri < windowSize; wri++ { // For every row in the window...
				for wci := 0; wci < windowSize; wci++ { // For every column in the window...
					currentWindowCellValue := data[wvsi+wri][whsi+wci] // Get the current window cell's value.
					if currentWindowCellValue > maxValueInWindow {     // If the current window cell's value is greater than the previously recorded greatest value...
						maxValueInWindow = currentWindowCellValue // Record this value as the new greatest value.
					}
				}
			}
			projectedLocation := wvsi / stride                                                      // This is the location the max value found should live at in the pooled data. This is needed because the pooled data array is smaller than the original array, and this is the adjustment to accommodate that.
			pooledData[projectedLocation] = append(pooledData[projectedLocation], maxValueInWindow) // After checking every cell in the window, append the greatest value found to the pooling data at its projected location.
		}
	}

	return pooledData
}

func leftPad(row []float64, lpad int) []float64 {
	for lpi := 0; lpi < lpad; lpi++ { // For every requested piece of left padding...
		row = append([]float64{0}, row...) // Prepend a zero to the current row.
	}
	return row
}

func rightPad(row []float64, rpad int) []float64 {
	for rpi := 0; rpi < rpad; rpi++ { // For every requested piece of right padding...
		row = append(row, 0) // Append a zero to the current row.
	}
	return row
}

func topPad(data [][]float64, tpad int) [][]float64 {
	dl := len(data[0])
	for tpi := 0; tpi < tpad; tpi++ { // For every requested piece of top padding...
		blankRow := jnet.Fill1d(dl, 0)                // Generate a blank row.
		data = append([][]float64{blankRow}, data...) // Prepend it to the top of the data.
	}
	return data
}

func bottomPad(data [][]float64, bpad int) [][]float64 {
	dl := len(data[0])
	for bpi := 0; bpi < bpad; bpi++ { // For every requested piece of bottom padding...
		blankRow := jnet.Fill1d(dl, 0) // Generate a blank row.
		data = append(data, blankRow)  // Append it to the bottom of the data.
	}
	return data
}

func splitPadding(pad int) (int, int) {
	pad1 := pad / 2
	pad2 := pad / 2
	if pad%2 != 0 { // If the padding can't be split evenly...
		pad2++ // Put the extra padding on pad2.
	}

	return pad1, pad2
}

func calculatePadding(length int, windowSize int, stride int) int {
	// Calculate which of the windowSize and stride is the greater value. This simplifies the for loop's condition below.
	greaterValue := int(math.Max(float64(windowSize), float64(stride)))

	maxIndex := 0 // Will store what the last index is to accommodate this windowSize and stride for this length.
	for ; maxIndex+greaterValue < length; maxIndex += stride {
		// For every window's start + the greater of the two values (windowSize and stride) which is still less than the length of the data...
		// This for loop intentionally does nothing, it is the mechanics of the for loop which do the calculation itself.
	}
	// Tack on the size of the final window to determine the final index needed.
	maxIndex += windowSize

	// Get the max of either the length or the recorded maxIndex, with certain values for windowSize and stride you can end up
	// with a maxIndex that is less than the length, which would lead to a negative padding. Correct this by setting it to
	// the length of the data if it is found to be less.
	maxIndex = int(math.Max(float64(length), float64(maxIndex)))

	// Padding is the recorded maxIndex needed minus the length of the data.
	padding := maxIndex - length

	return padding
}
