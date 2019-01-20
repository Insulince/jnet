package jnet

type HumanData struct {
	Data  [][]float64
	Truth []float64
}

func (hd *HumanData) ToTrainingData() (td *TrainingData) {
	td = &TrainingData{}

	qdr := len(hd.Data)
	for dri := 0; dri < qdr; dri++ {
		dr := hd.Data[dri]

		qd := len(dr)
		for di := 0; di < qd; di++ {
			datum := dr[di]

			td.Data = append(td.Data, datum)
		}
	}

	td.Truth = hd.Truth

	return td
}
