package trainer

import (
	"errors"
	"fmt"
	"github.com/Insulince/jnet/pkg/network"
	"math"
	"math/rand"
	"time"
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
}

func New(c Configuration, d Data) Trainer {
	// TODO(justin): Validate configuration?
	return Trainer{
		Configuration: c,
		Data:          d,
	}
}

func (d Data) shuffle() {
	for i := 0; i < len(d); i++ { // For every training datum in this training data...
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

	fmt.Println("Starting training process...")

	totalLoss, averageLoss, minMiniBatchLoss, maxMiniBatchLoss := 0.0, 0.0, float64(math.MaxInt32), float64(-math.MaxInt32)

	ti := 0
	for { // For every desired training iteration...
		miniBatch, err := t.Data.MiniBatch(t.Configuration.MiniBatchSize)
		if err != nil {
			return err
		}

		totalMiniBatchLoss := 0.0

		nw.ResetForPass(true)
		for _, td := range miniBatch {
			nw.ResetForPass(false)

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

		miniBatchLoss := totalMiniBatchLoss / float64(t.Configuration.MiniBatchSize) // Get the average loss across the whole mini batch.
		fmt.Printf("%3f ", miniBatchLoss)

		totalLoss += miniBatchLoss
		averageLoss = totalLoss / float64(ti) // TODO divide by zero????

		if miniBatchLoss > maxMiniBatchLoss {
			maxMiniBatchLoss = miniBatchLoss
		}
		if miniBatchLoss < minMiniBatchLoss {
			minMiniBatchLoss = miniBatchLoss
		}
		if (ti+1)%15 == 0 {
			fmt.Printf(" | %5f %5f %5f - %v\n", averageLoss, minMiniBatchLoss, maxMiniBatchLoss, ti)
		}

		nw.AdjustWeights(t.Configuration.LearningRate)

		select {
		case err := <-exit:
			return err
		default:
		}

		if averageLoss < t.Configuration.AverageLossCutoff {
			fmt.Printf("\nReached average loss cutoff limit, ending training process...\n")
			break
		}

		if minMiniBatchLoss < t.Configuration.MinLossCutoff {
			fmt.Printf("\nReached minimum loss cutoff limit, ending training process...\n")
			break
		}

		if ti > t.Configuration.MaxIterations {
			fmt.Printf("\nReached maximum iterations, ending training process...\n")
			break
		}

		ti++
	}

	fmt.Println("Training process ended.")

	return nil
}
