package hmm

import (
	"encoding/json"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	//	. "github.com/flyingyizi/tfutil/logicregression"
	//"gonum.org/v1/gonum/mat"
)

//HMM ..
type HMM struct {
	//N	 隐藏状态数目;Q={1,2,...,N}
	N int `json:"N"`
	//M	 观察符号数目; V={1,2,...,M}
	M int `json:"M"`

	//A	 状态转移矩阵A[1..N][1..N]. a[i][j] 是从t时刻状态i到t+1时刻状态j的转移概率
	A []float64 `json:"A"`
	//B  混淆矩阵B[1..N][1..M]. b[j][k]在状态j时观察到符合k的概率。
	B []float64 `json:"B"`
	// Pi[1..N] pi[i] is the initial state distribution.
	Pi []float64 `json:"pi"`
}

// Save saves model in the file
func (phmm *HMM) Save(path string) error {

	file, err := os.Create(path)
	if err == nil {
		encoder := json.NewEncoder(file)
		err = encoder.Encode(phmm)
	}
	file.Close()
	return err
}

// Load loads from the file
func (phmm *HMM) Load(path string) error {
	file, err := os.Open(path)
	if err == nil {
		decoder := json.NewDecoder(file)
		err = decoder.Decode(&phmm)
	}
	file.Close()

	return err
}

// GenSequenceArray 从当前HMM模型生成一个观测序列，以及对应的隐藏序列
//
//T is length of observation sequence
//O is the observation sequence O
//q is the state sequence q[1..T]
func (phmm *HMM) GenSequenceArray(seed int64, T int) (O, q []int) {
	// 算法：
	// 1) 按照初始状态分布pi产生状态i_1
	// 2)令t=1
	// 3)按照状态i_t的观测概率分布  b_{i_t} (k) 生成o_t
	// 4)按照状态i_t的状态转移概率分布产生状态i_{t+1}, i_{t+1}= 1,2,...N
	// 5)令t=t+1； 如果t<T, 转步(3); 否则，终止
	O, q = make([]int, T), make([]int, T)
	rand.Seed(seed) // hmmsetseed(seed);

	q[0] = phmm.genInitalState()
	O[0] = phmm.genSymbol(q[0])

	for t := 1; t < T; t++ {
		q[t] = phmm.genNextState(q[t-1])
		O[t] = phmm.genSymbol(q[t])
	}
	return
}

func (phmm *HMM) genInitalState() int {
	val := rand.Float64() // val = hmmgetrand();
	accum := 0.0
	qt := phmm.N
	for i := 0; i < phmm.N; i++ {
		if val < phmm.Pi[i]+accum {
			qt = i
			break
		} else {
			accum += phmm.Pi[i]
		}
	}

	return qt
}

func (phmm *HMM) genNextState(qt int) int {
	//phmm.A shape is N x N
	N := phmm.N

	val := rand.Float64() // val = hmmgetrand();

	accum := 0.0
	qNext := phmm.N
	for j := 0; j < phmm.N; j++ {
		if val < phmm.A[qt*N+j]+accum {
			qNext = j
			break
		} else {
			accum += phmm.A[qt*N+j]

		}
	}

	return qNext
}

func (phmm *HMM) genSymbol(qt int) int {
	//phmm.B shape is N x M
	M := phmm.M

	val := rand.Float64() // val = hmmgetrand();
	accum := 0.0
	ot := phmm.M
	for j := 0; j < phmm.M; j++ {
		if val < phmm.B[qt*M+j]+accum {
			ot = j
			break
		} else {
			accum += phmm.B[qt*M+j] //accum += phmm.B[qt][j];
		}
	}

	return ot
}

//Forward  HMM前向算法
// 解决HMM基本问题1：在给定模型λ=（A,B,Pi）与观测序列O。计算在给定模型λ下观测序列O出现的概率P(O|λ)
func (phmm *HMM) Forward(O []int) (pprob float64) {
	T := len(O)

	//alpha  shape is T x N
	alpha := Unflatten(T, phmm.N, make([]float64, T*phmm.N))
	A := mat.NewDense(phmm.N, phmm.N, phmm.A)
	B := mat.NewDense(phmm.N, phmm.M, phmm.B)

	/* 1. Initialization */
	//assign alpha[0][..]
	floats.MulTo(alpha[0],
		phmm.Pi, mat.Col(nil, O[0], B)) //联合概率

	/* 2. Induction */
	for t := 0; t < T-1; t++ { /* time index */

		for j := 0; j < phmm.N; j++ {
			//partial sum for *->j
			sum := floats.Dot(alpha[t], mat.Col(nil, j, A))
			alpha[t+1][j] = sum * (B.At(j, O[t+1]))
		}
	}

	/* 3. Termination */
	pprob = floats.Sum(alpha[T-1])

	return
}

//Viterbi  维特比算法
// 解决HMM基本问题2：对给定模型λ=（A,B,Pi）与观测序列O，计算观测序列O的隐隐马尔科夫状态的最短路径
func (phmm *HMM) Viterbi(O []int) (pprob float64, q []int) {
	T := len(O)
	//	delta  shape is T x N
	delta := mat.NewDense(T, phmm.N, nil)
	//psi  shape is T x N
	psi := mat.NewDense(T, phmm.N, nil)
	//q len is T
	q = make([]int, T)

	A := mat.NewDense(phmm.N, phmm.N, phmm.A)
	B := mat.NewDense(phmm.N, phmm.M, phmm.B)

	// 1. Initialization    t==0
	dst := make([]float64, phmm.N)
	floats.MulTo(dst, phmm.Pi, mat.Col(nil, O[0], B)) //联合概率
	delta.SetRow(0, dst)

	/* 2. Recursion */
	for t := 1; t < T; t++ { /* time index */
		for j := 0; j < phmm.N; j++ {
			floats.MulTo(dst, delta.RawRowView(t-1), mat.Col(nil, j, A))
			maxIdx := floats.MaxIdx(dst)

			delta.Set(t, j, dst[maxIdx]*(B.At(j, O[t])))
			psi.Set(t, j, float64(maxIdx))
		}
	}

	//  3. Termination
	q[T-1] = floats.MaxIdx(delta.RawRowView(T - 1))
	pprob = delta.At(T-1, q[T-1])

	//  4. Path (state sequence) backtracking
	for t := T - 2; t >= 0; t-- {
		q[t] = int(psi.At(t+1, q[t+1]))
	}
	return
}

//Unflatten  unflatten one dimensional to two-dimensional array
// with shape r by c
func Unflatten(r, c int, d []float64) [][]float64 {
	m := make([][]float64, r)
	for i := 0; i < r; i++ {
		m[i] = d[i*c : (i+1)*c]
	}
	return m
}

//Flatten flatten  two-dimensional array to one dimensional
func Flatten(f [][]float64) (r, c int, d []float64) {
	r = len(f)
	if r == 0 {
		panic("bad test: no row")
	}
	c = len(f[0])
	d = make([]float64, 0, r*c)
	for _, row := range f {
		if len(row) != c {
			panic("bad test: ragged input")
		}
		d = append(d, row...)
	}
	return r, c, d
}

//OSeq .. observation  sequence
type OSeq struct {
	//T  序列长度
	T int   `json:"T"`
	O []int `json:"O"`
}

// Save saves model in the file
func (f *OSeq) Save(path string) error {

	file, err := os.Create(path)
	if err == nil {
		encoder := json.NewEncoder(file)
		err = encoder.Encode(f)
	}
	file.Close()
	return err
}

// Load loads from the file
func (f *OSeq) Load(path string) error {
	file, err := os.Open(path)
	if err == nil {
		decoder := json.NewDecoder(file)
		err = decoder.Decode(&f)
	}
	file.Close()

	return err
}

// func main() {
// 	var h HMM

// 	h.M = 2
// 	h.N = 3
// 	h.A = []float64{0.333, 0.333, 0.333,
// 		0.333, 0.333, 0.333,
// 		0.333, 0.333, 0.333}

// 	h.B = []float64{0.5, 0.5,
// 		0.75, 0.25,
// 		0.25, 0.75}
// 	h.Pi = []float64{0.333, 0.333, 0.333}

// 	h.Save("testhmm.hmm")
// }

// func generateRangeNum(min, max int) int {
// 	rand.Seed(time.Now().Unix())
// 	randNum := rand.Intn(max-min) + min
// 	return randNum
// }
