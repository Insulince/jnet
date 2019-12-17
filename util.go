package jnet

func Duplicate4d(data [][][][]float64) [][][][]float64 {
	dupe := make([][][][]float64, len(data))
	for i := range data {
		dupe[i] = Duplicate3d(data[i])
	}
	return dupe
}

func Duplicate3d(data [][][]float64) [][][]float64 {
	dupe := make([][][]float64, len(data))
	for i := range data {
		dupe[i] = Duplicate2d(data[i])
	}
	return dupe
}

func Duplicate2d(data [][]float64) [][]float64 {
	dupe := make([][]float64, len(data))
	for i := range data {
		dupe[i] = Duplicate1d(data[i])
	}
	return dupe
}

func Duplicate1d(data []float64) []float64 {
	dupe := make([]float64, len(data))
	for i := range data {
		dupe[i] = data[i]
	}
	return dupe
}

func Fill4d(size1 int, size2 int, size3 int, size4 int, val float64) [][][][]float64 {
	data := make([][][][]float64, size1)

	for i := 0; i < size1; i++ {
		data[i] = Fill3d(size2, size3, size4, val)
	}

	return data
}

func Fill3d(size1 int, size2 int, size3 int, val float64) [][][]float64 {
	data := make([][][]float64, size1)

	for i := 0; i < size1; i++ {
		data[i] = Fill2d(size2, size3, val)
	}

	return data
}

func Fill2d(size1 int, size2 int, val float64) [][]float64 {
	data := make([][]float64, size1)

	for i := 0; i < size1; i++ {
		data[i] = Fill1d(size2, val)
	}

	return data
}

func Fill1d(size int, val float64) []float64 {
	data := make([]float64, size)

	for i := 0; i < size; i++ {
		data[i] = val
	}

	return data
}
