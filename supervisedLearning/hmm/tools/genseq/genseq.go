//给定模型λ=（A,B,Pi）,生成一个观测序列O
//example
// tools\genseq>go run genseq.go ../../t2.hmm
// RandomSeed: 1547040208
// {
//         "T": 10,
//         "O": [ 1,  0,  1, 0,  1, 0, 1, 0, 0, 1 ]
// }

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/flyingyizi/tfutil/supervisedLearning/hmm"
)

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: generate observation sequence based on λ=(A,B,Pi) defined by mod.hmm
	[-S:T:o] <mod.hmm>  
Options:
`)
	flag.PrintDefaults()
}

func main() {
	var h hmm.HMM /* the HMM */
	//flag
	seed := flag.Int64("S", 0, "random numer `seed`")
	T := flag.Int("T", 10, "`length` of observation sequence ")
	output := flag.String("o", "", "`filename` of file to store the observation sequence ")
	flag.Usage = usage

	flag.Parse()

	args := flag.Args()

	if len(args) != 1 {
		flag.Usage()
		return
	}

	/* read HMM file */
	h.Load(args[0])

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

	if len(*output) != 0 {
		observisionSeq.Save(*output)
	}

}
