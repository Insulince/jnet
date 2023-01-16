#!/usr/bin/env zsh

# Run the NetworkConvergence test.
go test "${GOPATH}/src/github.com/Insulince/jnet/pkg/network" -v -run Test_NetworkConverges
