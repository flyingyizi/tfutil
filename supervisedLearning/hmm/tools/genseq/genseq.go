//给定模型λ=（A,B,Pi）,生成一个观测序列O
//example tools\genseq>go run genseq.go
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"time"

	"github.com/flyingyizi/tfutil/supervisedLearning/hmm"
)

func usage() {
	// 	fmt.Fprintf(os.Stderr, `Usage error
	// 	T = length of sequence
	// 	S =  random number seed

	// Options:
	// `)
	flag.PrintDefaults()
}

func main() {
	var h hmm.HMM /* the HMM */
	//flag
	seed := flag.Int64("S", 0, "random numer `seed`")
	T := flag.Int("T", 10, "`length` of observation sequence ")
	flag.Usage = usage

	flag.Parse()
	// if flag.NFlag() != 2 {
	// 	flag.Usage()
	// 	return
	// }

	/* read HMM file */
	h.Load("../../testhmm.hmm")

	// 	/* length of observation sequence, T */

	if *seed == 0 {
		*seed = time.Now().Unix()
	}
	fmt.Println("RandomSeed:", *seed)
	// 	int	*O;	/* the observation sequence O[1..T]*/
	// 	int	*q; 	/* the state sequence q[1..T] */
	Ou, _ := h.GenSequenceArray(*seed, *T)

	observisionSeq := hmm.OSeq{T: *T, O: Ou}
	bData, _ := json.MarshalIndent(observisionSeq, "", "\t")
	fmt.Println(string(bData))
	observisionSeq.Save(fmt.Sprintf("../../test.%d.seq", *T))
}
