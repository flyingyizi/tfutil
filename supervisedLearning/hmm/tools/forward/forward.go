//go run forward.go  ../../t2.hmm  ../../t2.50.seq
// ------------------------------------
// Forward without scaling
// log prob(O| model) = -3.428056E+01

package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/flyingyizi/tfutil/supervisedLearning/hmm"
)

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: hmm Foward algorithm for computing the probabilty 
  of observing a sequence given a HMM model parameter and 
  observing sequence

  <model.hmm> <obs.seq> 

Options: NA
`)
	flag.PrintDefaults()
}

func main() {
	var h hmm.HMM     /* the HMM */
	var Oseq hmm.OSeq // obervision sequence
	flag.Usage = usage
	flag.Parse()

	if flag.NArg() != 2 {
		flag.Usage()
		return
	}

	args := flag.Args()
	hmmf, seqf := args[0], args[1]

	h.Load(hmmf)

	Oseq.Load(seqf)
	O, T := Oseq.O, Oseq.T /* observation sequence O[1..T] */
	if len(O) != T {
		panic("wrong observition length")
	}

	fmt.Println("------------------------------------")
	fmt.Println("Forward without scaling ")
	proba := h.Forward(O)
	fmt.Printf("log prob(O| model) = %e\n", math.Log(proba))
}
