package data

const (
	Max      V = 1.0
	Min      V = -1.0
	Midpoint   = (Max-Min)/2 + Min
)

// V is an abstraction over `float64`. In the future we could add receiver functions
// to it to make it more robust, for now its just an organizer.
// Think of "V" as being "Value".
type V float64

// D is an abstraction over `[]float64`, by being `[]V`. The same note applies about
// receiver functions.
// Think of "D" as being "Data".
type D []V

// T is an abstraction over `[]float64`, by being `D`. It is intended to contain data that is "true".
// This allows us to restrict and remind users that functions who accept this type are
// NOT looking for inputs, they are looking for data that is known to be true.
// Think of "T" as being "Truth".
type T D

// TrainingData contains an element `Data`, which contains sample inputs, and `Truth`, which contains
// the true output for this data. We can feed these objects into the network to train it.
type TrainingData struct {
	Data  D
	Truth T
}

// Loss is an abstraction over `float64`, by being `V`. This allows us to apply a similar reminder
// via typechecking as described in `T` above, except in places where a loss is expected.
type Loss V

// Gradient is an abstraction over `float64`, by being `V`. This allows us to apply a similar reminder
//// via typechecking as described in `T` above, except in places where a gradient is expected.
type Gradient V
