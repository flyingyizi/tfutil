// viterbi>go run vertebi.go  ../../t2.hmm  ../../t2.50.seq
// ------------------------------------
// Viterbi using direct probabilities
// Viterbi  MLE log prob = -4.091964E+01
// Optimal state sequence:
// T= 50
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/flyingyizi/tfutil/supervisedLearning/hmm"
)

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: hmm vertebi algorithm  given a HMM model parameter and 
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

	fmt.Printf("------------------------------------\n")
	fmt.Printf("Viterbi using direct probabilities\n")
	proba, q := h.Viterbi(O)

	fmt.Printf("Viterbi  MLE log prob = %E\n", math.Log(proba))
	fmt.Printf("Optimal state sequence:\n")

	fmt.Printf("T= %d\n", len(q))
	fmt.Println(q)

}
