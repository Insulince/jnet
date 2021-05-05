package network

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"github.com/pkg/errors"
)

type (
	Serializer interface {
		Serialize(Network) ([]byte, error)
		MustSerialize(Network) []byte
	}

	Deserializer interface {
		Deserialize([]byte) (Network, error)
		MustDeserialize([]byte) Network
	}

	Translator interface {
		Serializer
		Deserializer
	}

	TranslatorOption struct {
		Serialize   func(bs []byte) ([]byte, error)
		Deserialize func(bs []byte) ([]byte, error)
	}
)

func WithCompression() TranslatorOption {
	return TranslatorOption{
		Serialize: func(bs []byte) ([]byte, error) {
			var b bytes.Buffer
			gz := gzip.NewWriter(&b)
			if _, err := gz.Write(bs); err != nil {
				return nil, errors.Wrap(err, "gzip write")
			}
			if err := gz.Close(); err != nil {
				return nil, errors.Wrap(err, "gzip close")
			}
			return b.Bytes(), nil
		},
		Deserialize: func(bs []byte) ([]byte, error) {
			gz, err := gzip.NewReader(bytes.NewBuffer(bs))
			if err != nil {
				return nil, errors.Wrap(err, "gzip new reader")
			}
			if _, err := gz.Read(bs); err != nil {
				return nil, errors.Wrap(err, "gzip read")
			}
			if err := gz.Close(); err != nil {
				return nil, errors.Wrap(err, "gzip close")
			}
			return bs, nil
		},
	}
}

func WithBase64() TranslatorOption {
	return TranslatorOption{
		Serialize: func(bs []byte) ([]byte, error) {
			bs = []byte(base64.StdEncoding.EncodeToString(bs))
			return bs, nil
		},
		Deserialize: func(bs []byte) ([]byte, error) {
			var err error
			bs, err = base64.StdEncoding.DecodeString(string(bs))
			if err != nil {
				return nil, errors.Wrap(err, "base 64 decoding")
			}
			return bs, nil
		},
	}
}
