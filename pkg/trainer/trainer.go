package trainer

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/Insulince/jnet/pkg/network"
)

type Configuration struct {
	LearningRate      float64
	MiniBatchSize     int
	AverageLossCutoff float64
	MinLossCutoff     float64
	MaxIterations     int
	Timeout           time.Duration
}

type Datum struct {
	Data  []float64
	Truth []float64
}

type Data []Datum

type Trainer struct {
	Configuration
	Data
	Log io.Writer
}

// TODO(justin): Validate configuration?
func New(c Configuration, d Data, log io.Writer) Trainer {
	if log == nil {
		log = os.Stdout
	}
	return Trainer{
		Configuration: c,
		Data:          d,
		Log:           log,
	}
}

func (d Data) shuffle() {
	// For every training datum in this training data...
	for i := 0; i < len(d); i++ {
		r := rand.Intn(i + 1) // Select a random training datum index in [0, i]

		d[i], d[r] = d[r], d[i]
	}
}

func (d Data) MiniBatch(size int) (Data, error) {
	if size < 1 {
		return nil, errors.New("requested mini batch size must be at least 1")
	}
	if size > len(d) {
		return nil, errors.New("requested mini batch size larger than number of training datums")
	}

	d.shuffle()

	return d[:size], nil
}

var ErrTimedOut = errors.New("training process timed out")

func (t *Trainer) Train(nw network.Network) error {
	// TODO(justin): Make use of the timeout in the training configuration.

	exit := make(chan error)

	if t.Configuration.Timeout > 0 {
		go func() {
			time.Sleep(t.Configuration.Timeout)
			exit <- ErrTimedOut
		}()
	}

	_, _ = fmt.Fprintln(t.Log, "Starting training process...")

	totalLoss, averageLoss, minMiniBatchLoss, maxMiniBatchLoss := 0.0, 0.0, float64(math.MaxInt32), float64(-math.MaxInt32)

	ti := 0
	for { // For every desired training iteration...
		miniBatch, err := t.Data.MiniBatch(t.Configuration.MiniBatchSize)
		if err != nil {
			return err
		}

		totalMiniBatchLoss := 0.0

		nw.ResetFromBatch()
		for _, td := range miniBatch {
			nw.ResetFromPass()

			err := nw.ForwardPass(td.Data)
			if err != nil {
				return err
			}

			loss, err := nw.CalculateLoss(td.Truth)
			if err != nil {
				return err
			}

			totalMiniBatchLoss += loss

			err = nw.BackwardPass(td.Truth)
			if err != nil {
				return err
			}

			nw.RecordNudges()
		}

		// Get the average loss across the whole mini batch.
		miniBatchLoss := totalMiniBatchLoss / float64(t.Configuration.MiniBatchSize)
		_, _ = fmt.Fprintf(t.Log, "%3f ", miniBatchLoss)

		totalLoss += miniBatchLoss
		averageLoss = totalLoss / float64(ti) // TODO divide by zero????

		if miniBatchLoss > maxMiniBatchLoss {
			maxMiniBatchLoss = miniBatchLoss
		}
		if miniBatchLoss < minMiniBatchLoss {
			minMiniBatchLoss = miniBatchLoss
		}
		if (ti+1)%15 == 0 {
			_, _ = fmt.Fprintf(t.Log, " | %5f %5f %5f - %v\n", averageLoss, minMiniBatchLoss, maxMiniBatchLoss, ti)
		}

		nw.AdjustWeights(t.Configuration.LearningRate)

		select {
		case err := <-exit:
			return err
		default:
		}

		if averageLoss < t.Configuration.AverageLossCutoff {
			_, _ = fmt.Fprintf(t.Log, "\nReached average loss cutoff limit, ending training process...\n")
			break
		}

		if minMiniBatchLoss < t.Configuration.MinLossCutoff {
			_, _ = fmt.Fprintf(t.Log, "\nReached minimum loss cutoff limit, ending training process...\n")
			break
		}

		if ti > t.Configuration.MaxIterations {
			_, _ = fmt.Fprintf(t.Log, "\nReached maximum iterations, ending training process...\n")
			break
		}

		ti++
	}

	_, _ = fmt.Fprintln(t.Log, "Training process ended.")

	return nil
}
