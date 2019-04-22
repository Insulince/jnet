package jnet

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// TODO: This file. I mean c'mon, just look at it, my god...

type deserializationData struct {
	neuronMap         []int
	biases            [][]float64
	weights           [][][]float64
	inputLabels       []string
	outputLabels      []string
	timestamp         string
	predictionHistory []predictionHistory
	trainingHistory   []trainingHistory
}

func deserializeLengthError(section string, expected int, actual int) error {
	return errors.New(fmt.Sprintf("incorrect %v section: contains too many items, expected %v but found %v", section, expected, actual))
}

func deserializeTimeError(section string) error {
	return errors.New(fmt.Sprintf("incorrect %v section: invalid time format received, should be in RFC3339 format", section))
}

func Deserialize(networkString NetworkString) (nw *Network, err error) {
	var dd deserializationData

	lines := strings.Split(string(networkString), "\n")
	var items [][]string
	for _, line := range lines {
		items = append(items, strings.Split(line, separator))
	}

	versionDone, neuronMapDone, biasesDone, weightsDone, inputLabelsDone, outputLabelsDone, timestampDone, predictionHistoryDone := false, false, false, false, false, false, false, false
	expectedBiasLayer, expectedWeightLayer, expectedWeightNeuron := 1, 1, 0
	for _, line := range items {
		class := line[0]
		if class == versionKey {
			if versionDone {
				return nil, errors.New("attempting to deserialize version section but it has already been marked complete")
			}
			versionDone = true

			if len(line) != 2 {
				return nil, deserializeLengthError("version", 2, len(line))
			}

			data := line[1:]

			if data[0] != SerializerVersion {
				return nil, errors.New(fmt.Sprintf("invalid version section: incorrect version, expected %v but found %v", SerializerVersion, lines[1]))
			}

			continue
		}
		if class == neuronMapKey {
			if neuronMapDone {
				return nil, errors.New("attempting to deserialize neuron map section but it has already been marked complete")
			}
			neuronMapDone = true

			data := line[1:]

			var nm []int
			for _, item := range data {
				qn, err := strconv.Atoi(item)
				if err != nil {
					return nil, errors.New(fmt.Sprintf("%v while converting item in neuron map section", err))
				}
				nm = append(nm, qn)
			}

			dd.neuronMap = nm

			continue
		}
		if strings.HasPrefix(class, biasesKey) {
			if biasesDone {
				return nil, errors.New("attempting to deserialize biases section but it has already been marked complete")
			}

			key := line[0]
			keyItems := strings.Split(key, "_")
			if len(keyItems) != 3 {
				return nil, deserializeLengthError("biases (key)", 3, len(keyItems))
			}
			liStr := keyItems[2]
			li, err := strconv.Atoi(liStr)
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in biases section", err))
			}
			if li != expectedBiasLayer {
				return nil, errors.New(fmt.Sprintf("incorrect layer index found while processing biases, expected %v but found %v", expectedBiasLayer, li))
			}
			expectedBiasLayer++
			expectedLength := dd.neuronMap[li]

			data := line[1:]
			if len(data) != expectedLength {
				return nil, deserializeLengthError("biases (data)", expectedLength, len(data))
			}

			var biases []float64
			for _, item := range data {
				bias, err := strconv.ParseFloat(item, 64)
				if err != nil {
					return nil, errors.New(fmt.Sprintf("%v while converting item in biases section", err))
				}
				biases = append(biases, bias)
			}
			dd.biases = append(dd.biases, biases)

			continue
		}
		if strings.HasPrefix(class, weightsKey) {
			if weightsDone {
				return nil, errors.New("attempting to deserialize weights section but it has already been marked complete")
			}
			biasesDone = true

			key := line[0]
			keyItems := strings.Split(key, "_")
			if len(keyItems) != 5 {
				return nil, deserializeLengthError("weights (key)", 5, len(keyItems))
			}
			liStr := keyItems[2]
			li, err := strconv.Atoi(liStr)
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in weights section", err))
			}
			if li != expectedWeightLayer {
				return nil, errors.New(fmt.Sprintf("incorrect layer index found while processing weights, expected %v but found %v", expectedWeightLayer, li))
			}
			pli := li - 1

			niStr := keyItems[4]
			ni, err := strconv.Atoi(niStr)
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in weights section", err))
			}
			if ni != expectedWeightNeuron {
				return nil, errors.New(fmt.Sprintf("incorrect neuron index found while processing weights, expected %v but found %v", expectedWeightNeuron, ni))
			}
			if expectedWeightNeuron == 0 {
				dd.weights = append(dd.weights, [][]float64{})
			}
			expectedWeightNeuron++
			if expectedWeightNeuron == dd.neuronMap[expectedWeightLayer] {
				expectedWeightNeuron = 0
				expectedWeightLayer++
			}

			expectedLength := dd.neuronMap[pli]

			data := line[1:]
			if len(data) != expectedLength {
				return nil, deserializeLengthError("weights (data)", expectedLength, len(data))
			}

			var weights []float64
			for _, item := range data {
				bias, err := strconv.ParseFloat(item, 64)
				if err != nil {
					return nil, errors.New(fmt.Sprintf("%v while converting item in weights section", err))
				}
				weights = append(weights, bias)
			}
			dd.weights[pli] = append(dd.weights[pli], weights)

			continue
		}
		if class == inputLabelsKey {
			if inputLabelsDone {
				return nil, errors.New("attempting to deserialize input labels section but it has already been marked complete")
			}
			weightsDone = true
			inputLabelsDone = true

			data := line[1:]
			if len(data) != dd.neuronMap[0] {
				return nil, deserializeLengthError("input labels", dd.neuronMap[0], len(data))
			}

			dd.inputLabels = data

			continue
		}
		if class == outputLabelsKey {
			if outputLabelsDone {
				return nil, errors.New("attempting to deserialize output labels section but it has already been marked complete")
			}
			outputLabelsDone = true

			data := line[1:]
			if len(data) != dd.neuronMap[len(dd.neuronMap)-1] {
				return nil, deserializeLengthError("output labels", dd.neuronMap[len(dd.neuronMap)-1], len(data))
			}

			dd.outputLabels = data

			continue
		}
		if class == timestampKey {
			if timestampDone {
				return nil, errors.New("attempting to deserialize timestamp section but it has already been marked complete")
			}
			timestampDone = true

			data := line[1:]
			if len(data) != 1 {
				return nil, deserializeLengthError("timestamp", 1, len(data))
			}

			_, err = time.Parse(time.RFC3339, data[0])
			if err != nil {
				return nil, deserializeTimeError("timestamp")
			}

			dd.timestamp = data[0]

			continue
		}
		if class == predictionHistoryKey {
			if predictionHistoryDone {
				return nil, errors.New("attempting to deserialize prediction history section but it has already been marked complete")
			}

			data := line[1:]
			if len(data) != 4 {
				if len(data) == 1 {
					if data[0] == noDataIdentifier {
						continue
					}
				}
				return nil, deserializeLengthError("prediction history", 4, len(data))
			}

			inputStrs := strings.Split(data[2], ",")
			var inputs []float64
			for _, is := range inputStrs {
				input, err := strconv.ParseFloat(is, 64)
				if err != nil {
					return nil, errors.New(fmt.Sprintf("%v while converting item in prediciton history section", err))
				}
				inputs = append(inputs, input)
			}

			outputStrs := strings.Split(data[3], ",")
			var outputs []float64
			for _, os := range outputStrs {
				output, err := strconv.ParseFloat(os, 64)
				if err != nil {
					return nil, errors.New(fmt.Sprintf("%v while converting item in prediction history section", err))
				}
				outputs = append(outputs, output)
			}

			_, err = time.Parse(time.RFC3339, data[1])
			if err != nil {
				return nil, deserializeTimeError("prediction history")
			}

			ph := predictionHistory{
				Prediction: data[0],
				Timestamp:  data[1],
				Input:      inputs,
				Output:     outputs,
			}

			dd.predictionHistory = append(dd.predictionHistory, ph)

			continue
		}
		if class == trainingHistoryKey {
			predictionHistoryDone = true

			data := line[1:]
			if len(data) != 7 {
				if len(data) == 1 {
					if data[0] == noDataIdentifier {
						continue
					}
				}
				return nil, deserializeLengthError("training history", 7, len(data))
			}

			lr, err := strconv.ParseFloat(data[0], 64)
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in training history section", err))
			}
			iter, err := strconv.Atoi(data[1])
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in training history section", err))
			}
			mbs, err := strconv.Atoi(data[2])
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in training history section", err))
			}
			alc, err := strconv.ParseFloat(data[3], 64)
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in training history section", err))
			}
			dss, err := strconv.Atoi(data[4])
			if err != nil {
				return nil, errors.New(fmt.Sprintf("%v while converting item in training history section", err))
			}

			_, err = time.Parse(time.RFC3339, data[5])
			if err != nil {
				return nil, deserializeTimeError("training history")
			}
			_, err = time.Parse(time.RFC3339, data[6])
			if err != nil {
				return nil, deserializeTimeError("training history")
			}

			tc := &TrainingConfiguration{
				LearningRate:      lr,
				Iterations:        iter,
				MiniBatchSize:     mbs,
				AverageLossCutoff: alc,
			}
			th := trainingHistory{
				TrainingConfiguration: tc,
				DataSetSize:           dss,
				Start:                 data[5],
				Finish:                data[6],
			}

			dd.trainingHistory = append(dd.trainingHistory, th)

			continue
		}

		return nil, errors.New(fmt.Sprintf("unrecognized line encountered: %v", strings.Join(line, separator)))
	}

	if !versionDone || !neuronMapDone || !biasesDone || !weightsDone || !inputLabelsDone || !outputLabelsDone || !predictionHistoryDone {
		return nil, errors.New("not all sections were marked complete during deserialization process")
	}

	// CREATE NETWORK
	nw = &Network{}

	// BUILD LAYERS & INITIAL CONNECTIONS
	qnm := len(dd.neuronMap)
	for nmi := 0; nmi < qnm; nmi++ { // For every quantity of neuron in the neuron map...
		qn := dd.neuronMap[nmi]

		if nmi == 0 {
			nw.layers = append(nw.layers, newLayer(int(qn), nil))
		} else {
			pl := nw.layers[nmi-1]
			nw.layers = append(nw.layers, newLayer(int(qn), pl))
		}
	}

	// SETUP WEIGHTS AND BIASES
	ql := len(nw.layers)
	for li := 1; li < ql; li++ { // For every layer in the network...
		l := nw.layers[li]

		// BIASES
		qn := len(l.neurons)
		for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
			n := l.neurons[ni]

			n.bias = dd.biases[li-1][ni]
		}

		// WEIGHTS
		for ni := 0; ni < qn; ni++ { // For every neuron in this layer...
			n := l.neurons[ni]

			qc := len(n.connections)
			for ci := 0; ci < qc; ci++ { // For every connection from this neuron to the previous layer...
				c := n.connections[ci]

				c.weight = dd.weights[li-1][ni][ci]
			}
		}
	}

	// SETUP INPUT LABELS
	qil := len(dd.inputLabels)
	if qil != dd.neuronMap[0] {
		return nil, errors.New("cannot deserialize, number of labels and input neurons do not match")
	}
	fl := nw.layers[0]
	for il := 0; il < qil; il++ { // For every input label...
		label := dd.inputLabels[il]

		fl.neurons[il].label = label
	}

	// SETUP OUTPUT LABELS
	qol := len(dd.outputLabels)
	if qol != dd.neuronMap[qnm-1] {
		return nil, errors.New("cannot deserialize, number of labels and output neurons do not match")
	}
	ll := nw.layers[ql-1]
	for ol := 0; ol < qol; ol++ { // For every output label...
		label := dd.outputLabels[ol]

		ll.neurons[ol].label = label
	}

	// SETUP TIMESTAMP
	nw.Metadata.timestamp = dd.timestamp

	// SETUP PREDICTION HISTORY
	nw.Metadata.PredictionHistory = dd.predictionHistory

	// SETUP TRAINING HISTORY
	nw.Metadata.TrainingHistory = dd.trainingHistory

	return nw, nil
}
