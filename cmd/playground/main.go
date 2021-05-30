package main

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/pkg/errors"

	"github.com/Insulince/jnet/pkg/network/networkspb"
)

const (
	candidates = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789)!@#$%^&*(-=[]\\;',./`~_+{}|:\"<>?"
)

type (
	Builder struct {
		maxFloatInt int

		minFloatSliceLength  int
		floatSliceEscapeRate float64

		minStringLength  int
		stringEscapeRate float64

		minLayersLength  int
		layersEscapeRate float64

		minNeuronsLength  int
		neuronsEscapeRate float64

		minConnectionsLength  int
		connectionsEscapeRate float64
	}
)

func init() {
	rand.Seed(time.Now().UnixNano())
	gob.Register(&networkspb.Network{})
}

func (b Builder) randFloatSlice() []float64 {
	var fs []float64

	for i := 0; i < b.minFloatSliceLength || rand.Float64() < b.floatSliceEscapeRate; i++ {
		f := b.randFloat()
		fs = append(fs, f)
	}

	return fs
}

func (b Builder) randFloat() float64 {
	var v float64

	c := rand.Float64()

	switch true {
	case c > 0 && c < 0.333333:
		// integer
		i := float64(rand.Intn(b.maxFloatInt))
		v = i
	case c > 0.333333 && c < 0.666666:
		// float
		f := rand.Float64()
		v = f
	case c > 0.666666 && c < 1:
		// float and integer
		i := float64(rand.Intn(b.maxFloatInt))
		f := rand.Float64()
		v = i + f
	}

	switch true {
	case c < 0.5:
		// negative
		v *= -1
	default:
		// positive
		v *= 1
	}

	return v
}

func (b Builder) randString() string {
	var s string

	for i := 0; i < b.minStringLength || rand.Float64() < b.stringEscapeRate; i++ {
		c := b.randCharacter()
		s += string(c)
	}

	return s
}

func (b Builder) randCharacter() uint8 {
	c := candidates[rand.Intn(len(candidates))]

	return c
}

func (b Builder) randNetwork() *networkspb.Network {
	nw := &networkspb.Network{}

	nw.Layers = b.randLayers()
	nw.ActivationFunctionName = b.randString()

	return nw
}

func (b Builder) randLayers() []*networkspb.Layer {
	var ls []*networkspb.Layer

	for i := 0; i < b.minLayersLength || rand.Float64() < b.layersEscapeRate; i++ {
		l := b.randLayer()
		ls = append(ls, l)
	}

	return ls
}

func (b Builder) randLayer() *networkspb.Layer {
	l := &networkspb.Layer{}

	l.Neurons = b.randNeurons()

	return l
}

func (b Builder) randNeurons() []*networkspb.Neuron {
	var ns []*networkspb.Neuron

	for i := 0; i < b.minNeuronsLength || rand.Float64() < b.neuronsEscapeRate; i++ {
		n := b.randNeuron()
		ns = append(ns, n)
	}

	return ns
}

func (b Builder) randNeuron() *networkspb.Neuron {
	n := &networkspb.Neuron{}

	n.Connections = b.randConnections()
	n.Label = b.randString()
	n.Bias = b.randFloat()
	//n.Value = b.randFloat()
	//n.WSum = b.randFloat()
	//n.DLossDValue = b.randFloat()
	//n.DLossDBias = b.randFloat()
	//n.DValueDNet = b.randFloat()
	//n.DNetDBias = b.randFloat()
	//n.BiasNudges = b.randFloatSlice()

	return n
}

func (b Builder) randConnections() []*networkspb.Connection {
	var cs []*networkspb.Connection

	for i := 0; i < b.minConnectionsLength || rand.Float64() < b.connectionsEscapeRate; i++ {
		c := b.randConnection()
		cs = append(cs, c)
	}

	return cs
}

func (b Builder) randConnection() *networkspb.Connection {
	c := &networkspb.Connection{}

	c.Weight = b.randFloat()
	//c.DNetDWeight = b.randFloat()
	//c.DLossDWeight = b.randFloat()
	//c.DNetDPrevValue = b.randFloat()
	//c.WeightNudges = b.randFloatSlice()

	return c
}

func main() {
	//b := Builder{
	//	maxFloatInt:           100,
	//	minFloatSliceLength:   16,
	//	floatSliceEscapeRate:  0.0625,
	//	minStringLength:       20,
	//	stringEscapeRate:      0.05,
	//	minLayersLength:       2,
	//	layersEscapeRate:      0.2,
	//	minNeuronsLength:      16,
	//	neuronsEscapeRate:     0.0625,
	//	minConnectionsLength:  16,
	//	connectionsEscapeRate: 0.0625,
	//}
	b := Builder{
		maxFloatInt:           100,
		minFloatSliceLength:   576,
		floatSliceEscapeRate:  0.00625,
		minStringLength:       20,
		stringEscapeRate:      0.005,
		minLayersLength:       5,
		layersEscapeRate:      0.05,
		minNeuronsLength:      576,
		neuronsEscapeRate:     0.00625,
		minConnectionsLength:  576,
		connectionsEscapeRate: 0.00625,
	}
	nw := b.randNetwork()

	display(nw)

	fmt.Println()

	measurements := map[string]int{
		"raw":                 measureRaw(nw),
		"raw compressed":      measureRawCompressed(nw),
		"raw 64":              measureRaw64(nw),
		"raw compressed 64":   measureRawCompressed64(nw),
		"raw 64 compressed":   measureRaw64Compressed(nw),
		"json":                measureJson(nw),
		"json compressed":     measureJsonCompressed(nw),
		"json 64":             measureJson64(nw),
		"json compressed 64":  measureJsonCompressed64(nw),
		"json 64 compressed":  measureJson64Compressed(nw),
		"gob":                 measureGob(nw),
		"gob compressed":      measureGobCompressed(nw),
		"gob 64":              measureGob64(nw),
		"gob compressed 64":   measureGobCompressed64(nw),
		"gob 64 compressed":   measureGob64Compressed(nw),
		"proto":               measureProto(nw),
		"proto compressed":    measureProtoCompressed(nw),
		"proto 64":            measureProto64(nw),
		"proto compressed 64": measureProtoCompressed64(nw),
		"proto 64 compressed": measureProto64Compressed(nw),

		//"proto compressed 2x":  measureProtoCompressed2x(nw),
		//"proto compressed 4x":  measureProtoCompressed4x(nw),
		//"proto compressed 8x":  measureProtoCompressed8x(nw),
		//"proto compressed 16x": measureProtoCompressed16x(nw),
	}

	fmt.Println()

	smallestMeasurement := math.MaxInt64
	smallestId := ""
	for id, measurement := range measurements {
		if measurement < smallestMeasurement {
			smallestMeasurement = measurement
			smallestId = id
		}
	}
	fmt.Printf("%s (%2.3f%% the size of 'raw')", smallestId, 100*float64(smallestMeasurement)/float64(measurements["raw"]))
}

func display(v interface{}) {
	fmt.Println(v)
}

func size(v interface{}) int {
	var sz int
	if sv, ok := v.(string); ok {
		sz = len(sv)
	} else if bsv, ok := v.([]byte); ok {
		sz = len(bsv)
	} else {
		sv := fmt.Sprintf("%v", v)
		sz = len(sv)
	}
	u := 0
	for sz > 1000 {
		sz /= 1000
		u++
	}
	var unit string
	switch u {
	case 0:
		unit = "B"
	case 1:
		unit = "kB"
	case 2:
		unit = "MB"
	case 3:
		unit = "GB"
	case 4:
		unit = "TB"
	default:
		panic(errors.New("you are encoding literally more than 1000 TB of data, what are you doing bruh"))
	}
	fmt.Printf("%v %s\n", sz, unit)
	return sz
}

func rawMarshal(nw *networkspb.Network) []byte {
	return []byte(fmt.Sprint(nw))
}

func jsonMarshal(nw *networkspb.Network) []byte {
	bs, err := json.Marshal(nw)
	if err != nil {
		panic(errors.Wrap(err, "json marshaling"))
	}
	return bs
}

func gobMarshal(nw *networkspb.Network) []byte {
	var b bytes.Buffer
	err := gob.NewEncoder(&b).Encode(nw)
	if err != nil {
		panic(errors.Wrap(err, "gob marshaling"))
	}
	return b.Bytes()
}

func protoMarshal(nw *networkspb.Network) []byte {
	bs, err := proto.Marshal(nw)
	if err != nil {
		panic(errors.Wrap(err, "proto marshaling"))
	}
	return bs
}

func compress(bs []byte) []byte {
	var b bytes.Buffer
	gz := gzip.NewWriter(&b)
	_, err := gz.Write(bs)
	if err != nil {
		panic(errors.Wrap(err, "gzip write"))
	}
	err = gz.Close()
	if err != nil {
		panic(errors.Wrap(err, "gzip close"))
	}
	return b.Bytes()
}

func b64(bs []byte) []byte {
	return []byte(base64.StdEncoding.EncodeToString(bs))
}

func measureRaw(nw *networkspb.Network) int {
	fmt.Print("raw: ")
	bs := rawMarshal(nw)
	return size(bs)
}

func measureRawCompressed(nw *networkspb.Network) int {
	fmt.Print("raw compressed: ")
	bs := rawMarshal(nw)
	bs = compress(bs)
	return size(bs)
}

func measureRaw64(nw *networkspb.Network) int {
	fmt.Print("raw 64: ")
	bs := rawMarshal(nw)
	bs = b64(bs)
	return size(bs)
}

func measureRawCompressed64(nw *networkspb.Network) int {
	fmt.Print("raw compressed 64: ")
	bs := rawMarshal(nw)
	bs = compress(bs)
	bs = b64(bs)
	return size(bs)
}

func measureRaw64Compressed(nw *networkspb.Network) int {
	fmt.Print("raw 64 compressed: ")
	bs := rawMarshal(nw)
	bs = b64(bs)
	bs = compress(bs)
	return size(bs)
}

func measureJson(nw *networkspb.Network) int {
	fmt.Print("json: ")
	bs := jsonMarshal(nw)
	return size(bs)
}

func measureJsonCompressed(nw *networkspb.Network) int {
	fmt.Print("json compressed: ")
	bs := jsonMarshal(nw)
	bs = compress(bs)
	return size(bs)
}

func measureJson64(nw *networkspb.Network) int {
	fmt.Print("json 64: ")
	bs := jsonMarshal(nw)
	bs = b64(bs)
	return size(bs)
}

func measureJsonCompressed64(nw *networkspb.Network) int {
	fmt.Print("json compressed 64: ")
	bs := jsonMarshal(nw)
	bs = compress(bs)
	bs = b64(bs)
	return size(bs)
}

func measureJson64Compressed(nw *networkspb.Network) int {
	fmt.Print("json 64 compressed: ")
	bs := jsonMarshal(nw)
	bs = b64(bs)
	bs = compress(bs)
	return size(bs)
}

func measureGob(nw *networkspb.Network) int {
	fmt.Print("gob: ")
	bs := gobMarshal(nw)
	return size(bs)
}

func measureGobCompressed(nw *networkspb.Network) int {
	fmt.Print("gob compressed: ")
	bs := gobMarshal(nw)
	bs = compress(bs)
	return size(bs)
}

func measureGob64(nw *networkspb.Network) int {
	fmt.Print("gob 64: ")
	bs := gobMarshal(nw)
	bs = b64(bs)
	return size(bs)
}

func measureGobCompressed64(nw *networkspb.Network) int {
	fmt.Print("gob compressed 64: ")
	bs := gobMarshal(nw)
	bs = compress(bs)
	bs = b64(bs)
	return size(bs)
}

func measureGob64Compressed(nw *networkspb.Network) int {
	fmt.Print("gob 64 compressed: ")
	bs := gobMarshal(nw)
	bs = b64(bs)
	bs = compress(bs)
	return size(bs)
}

func measureProto(nw *networkspb.Network) int {
	fmt.Print("proto: ")
	bs := protoMarshal(nw)
	return size(bs)
}

func measureProtoCompressed(nw *networkspb.Network) int {
	fmt.Print("proto compressed: ")
	bs := protoMarshal(nw)
	bs = compress(bs)
	return size(bs)
}

func measureProto64(nw *networkspb.Network) int {
	fmt.Print("proto 64: ")
	bs := protoMarshal(nw)
	bs = b64(bs)
	return size(bs)
}

func measureProtoCompressed64(nw *networkspb.Network) int {
	fmt.Print("proto compressed 64: ")
	bs := protoMarshal(nw)
	bs = compress(bs)
	bs = b64(bs)
	return size(bs)
}

func measureProto64Compressed(nw *networkspb.Network) int {
	fmt.Print("proto 64 compressed: ")
	bs := protoMarshal(nw)
	bs = b64(bs)
	bs = compress(bs)
	return size(bs)
}

func measureProtoCompressed2x(nw *networkspb.Network) int {
	fmt.Print("proto compressed 2x: ")
	bs := protoMarshal(nw)
	bs = compress(bs)
	bs = compress(bs)
	return size(bs)
}

func measureProtoCompressed4x(nw *networkspb.Network) int {
	fmt.Print("proto compressed 4x: ")
	bs := protoMarshal(nw)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	return size(bs)
}

func measureProtoCompressed8x(nw *networkspb.Network) int {
	fmt.Print("proto compressed 8x: ")
	bs := protoMarshal(nw)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	return size(bs)
}

func measureProtoCompressed16x(nw *networkspb.Network) int {
	fmt.Print("proto compressed 16x: ")
	bs := protoMarshal(nw)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	bs = compress(bs)
	return size(bs)
}
