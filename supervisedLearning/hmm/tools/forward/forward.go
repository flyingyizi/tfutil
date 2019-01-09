package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/flyingyizi/tfutil/supervisedLearning/hmm"
)

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: testfor <model.hmm> <obs.seq> 
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

	// alpha = dmatrix(1, T, 1, hmm.N)
	// scale = dvector(1, T)

	fmt.Println("------------------------------------")
	fmt.Println("Forward without scaling ")
	_, proba := h.Forward(O)
	fmt.Printf("log prob(O| model) = %v\n", math.Log(proba))

	// printf("------------------------------------\n")
	// printf("Forward with scaling \n")

	// ForwardWithScale(&hmm, T, O, alpha, scale, &logproba)

	// fprintf(stdout, "log prob(O| model) = %E\n", logproba)
	// printf("------------------------------------\n")
	// printf("The two log probabilites should identical \n")
	// printf("(within numerical precision). When observation\n")
	// printf("sequence is very large, use scaling. \n")

	// free_ivector(O, 1, T)
	// free_dmatrix(alpha, 1, T, 1, hmm.N)
	// free_dvector(scale, 1, T)
	// FreeHMM(&hmm)
}
