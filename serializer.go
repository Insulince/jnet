package jnet

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	SerializerVersion = "2.0.0"

	separator = "|"

	versionKey           = "VERSION"
	neuronMapKey         = "NEURON_MAP"
	biasesKey            = "BIASES"
	weightsKey           = "WEIGHTS"
	inputLabelsKey       = "INPUT_LABELS"
	nameKey              = "NAME"
	descriptionKey       = "DESCRIPTION"
	timestampKey         = "TIMESTAMP"
	outputLabelsKey      = "OUTPUT_LABELS"
	predictionHistoryKey = "PREDICTION_HISTORY"
	trainingHistoryKey   = "TRAINING_HISTORY"
	logKey               = "LOG"

	layerIdentifier  = "LAYER"
	neuronIdentifier = "NEURON"

	noDataIdentifier = "<no-data>"
)

// VERSION

func serializeVersionLine() string {
	return fmt.Sprintf("%v%v%v", versionKey, separator, SerializerVersion)
}

// NEURON MAP

func serializeNeuronMapLine(layers []*layer) string {
	var sizes []string
	for _, l := range layers {
		sizes = append(sizes, strconv.Itoa(len(l.neurons)))
	}
	return fmt.Sprintf("%v%v%v", neuronMapKey, separator, strings.Join(sizes, separator))
}

// BIASES

func serializeBiasesLines(biasedLayers []*layer) string {
	var biasesLines []string
	for bli, bl := range biasedLayers {
		biasesLines = append(biasesLines, serializeBiasLine(bli, bl))
	}
	return strings.Join(biasesLines, "\n")
}

func serializeBiasLine(bli int, bl *layer) string {
	var biases []string
	for _, n := range bl.neurons {
		biases = append(biases, fmt.Sprintf("%v", n.bias))
	}
	return fmt.Sprintf("%v%v%v", generateBiasesLabel(bli), separator, strings.Join(biases, separator))
}

func generateBiasesLabel(bli int) string {
	return fmt.Sprintf("%v_%v", biasesKey, generateLayerLabel(bli+1))
}

// WEIGHTS

func serializeWeightsLines(weightedLayers []*layer) string {
	var weightsLines []string
	for wli, wl := range weightedLayers {
		weightsLines = append(weightsLines, serializeWeightLinesPerLayer(wli, wl))
	}
	return strings.Join(weightsLines, "\n")
}

func serializeWeightLinesPerLayer(wli int, wl *layer) string {
	var weightLinesPerLayer []string
	for ni, n := range wl.neurons {
		weightLinesPerLayer = append(weightLinesPerLayer, serializeWeightLine(wli, n, ni))
	}
	return strings.Join(weightLinesPerLayer, "\n")
}

func serializeWeightLine(wli int, n *neuron, ni int) string {
	var weights []string
	for _, c := range n.connections {
		weights = append(weights, fmt.Sprintf("%v", c.weight))
	}
	return fmt.Sprintf("%v%v%v", generateWeightsLabel(wli, ni), separator, strings.Join(weights, separator))
}

func generateWeightsLabel(wli int, ni int) string {
	return fmt.Sprintf("%v_%v_%v", weightsKey, generateLayerLabel(wli+1), generateNeuronLabel(ni))
}

// INPUT LABELS

func serializeInputLabelsLine(fl *layer) string {
	var inputLabels []string
	for _, n := range fl.neurons {
		inputLabels = append(inputLabels, n.label)
	}
	return fmt.Sprintf("%v%v%v", inputLabelsKey, separator, strings.Join(inputLabels, separator))
}

// OUTPUT LABELS

func serializeOutputLabelsLine(ll *layer) string {
	var outputLabels []string
	for _, n := range ll.neurons {
		outputLabels = append(outputLabels, n.label)
	}
	return fmt.Sprintf("%v%v%v", outputLabelsKey, separator, strings.Join(outputLabels, separator))
}

// NAME

func serializeNameLine(name string) string {
	return fmt.Sprintf("%v%v%v", nameKey, separator, name)
}

// DESCRIPTION

func serializeDescriptionLine(description string) string {
	return fmt.Sprintf("%v%v%v", descriptionKey, separator, description)
}

// TIMESTAMP

func serializeTimestampLine(timestamp string) string {
	return fmt.Sprintf("%v%v%v", timestampKey, separator, timestamp)
}

// PREDICTION HISTORY

func serializePredictionHistoryLines(predictionHistory []predictionHistory) string {
	var predictionHistoryLines []string
	for _, ph := range predictionHistory {
		predictionHistoryLines = append(predictionHistoryLines, serializePredictionHistoryLine(ph))
	}
	if len(predictionHistory) == 0 {
		return strings.Join([]string{predictionHistoryKey, noDataIdentifier}, separator)
	}
	return strings.Join(predictionHistoryLines, "\n")
}

func serializePredictionHistoryLine(ph predictionHistory) string {
	return fmt.Sprintf(
		"%v%v%v%v%v%v%v%v%v",
		predictionHistoryKey, separator,
		ph.Prediction, separator,
		ph.Timestamp, separator,
		serializePredictionHistoryInputs(ph.Input), separator,
		serializePredictionHistoryOutputs(ph.Output),
	)
}

func serializePredictionHistoryInputs(inputs []float64) string {
	var is []string
	for _, i := range inputs {
		is = append(is, fmt.Sprintf("%v", i))
	}
	return strings.Join(is, ",")

}

func serializePredictionHistoryOutputs(outputs []float64) string {
	var os []string
	for _, o := range outputs {
		os = append(os, fmt.Sprintf("%v", o))
	}
	return strings.Join(os, ",")
}

// TRAINING HISTORY

func serializeTrainingHistoryLines(trainingHistory []trainingHistory) string {
	var trainingHistoryLines []string
	for _, ph := range trainingHistory {
		trainingHistoryLines = append(trainingHistoryLines, serializeTrainingHistoryLine(ph))
	}
	if len(trainingHistory) == 0 {
		return strings.Join([]string{trainingHistoryKey, noDataIdentifier}, separator)
	}
	return strings.Join(trainingHistoryLines, "\n")
}

func serializeTrainingHistoryLine(th trainingHistory) string {
	return fmt.Sprintf(
		"%v%v%v%v%v%v%v%v%v%v%v%v%v%v%v",
		trainingHistoryKey, separator,
		th.LearningRate, separator,
		th.Iterations, separator,
		th.MiniBatchSize, separator,
		th.AverageLossCutoff, separator,
		th.DataSetSize, separator,
		th.Start, separator,
		th.Finish,
	)
}

// LOG

func serializeLogLines(log string) string {
	var logLines []string
	rawLogLines := strings.Split(log, "\n")
	for _, rll := range rawLogLines {
		logLines = append(logLines, serializeLogLine(rll))
	}
	return strings.Join(logLines, "\n")
}

func serializeLogLine(rawLogLine string) string {
	return fmt.Sprintf("%v%v%v", logKey, separator, rawLogLine)
}

// SHARED

func generateLayerLabel(li int) string {
	return fmt.Sprintf("%v_%v", layerIdentifier, li)
}

func generateNeuronLabel(ni int) string {
	return fmt.Sprintf("%v_%v", neuronIdentifier, ni)
}

// SERIALIZE

func (nw *Network) Serialize() NetworkString {
	fl := nw.layers[0]
	ll := nw.layers[len(nw.layers)-1]
	lines := []string{
		serializeVersionLine(),
		serializeNeuronMapLine(nw.layers),
		serializeBiasesLines(nw.layers[1:]),
		serializeWeightsLines(nw.layers[1:]),
		serializeInputLabelsLine(fl),
		serializeOutputLabelsLine(ll),
		serializeNameLine(nw.Metadata.Name),
		serializeDescriptionLine(nw.Metadata.Description),
		serializeTimestampLine(nw.Metadata.timestamp),
		serializePredictionHistoryLines(nw.Metadata.PredictionHistory),
		serializeTrainingHistoryLines(nw.Metadata.TrainingHistory),
		serializeLogLines(nw.Metadata.Log),
	}
	return NetworkString(strings.Join(lines, "\n"))
}
