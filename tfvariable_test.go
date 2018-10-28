package tfutil

import (
	"fmt"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

//测试tf.ArgMax，返回行或列的最大值下标向量
func TestArgMax(t *testing.T) {

	// import tensorflow as tf
	// a = tf.constant( [[1, 2, 3], [4, 5, 6]] )
	// #a = tf.constant( [[2, 2, 2], [2, 2, 2]] )
	// b=tf.argmax(input=a,axis=0)
	// c=tf.argmax(input=a,dimension=1)   #此处用dimesion或用axis是一样的
	// # Start tf session
	// sess = tf.Session()
	// print(sess.run(b))
	// #[1 1 1]
	// print(sess.run(c))
	// #[2 2]

	var (
		root       = op.NewScope()
		axis_row   = op.Const(root.SubScope("input"), int32(1)) //axis：0表示按列，1表示按行
		axis_colum = op.Const(root.SubScope("input"), int32(0))
	)

	testdata := [][][]int32{{{1, 2, 3}, {4, 5, 6}}, {{2, 2, 2}, {2, 2, 2}}}

	row_outputlist := make([]tf.Output, 0)
	colum_outputlist := make([]tf.Output, 0)

	fmt.Println("orig testdata:")
	for i, test := range testdata {
		fmt.Println("	", i, " testdata:", test)
		x := op.ArgMax(root.SubScope("input"), op.Const(root.SubScope("input"), test), axis_row)
		row_outputlist = append(row_outputlist, x)
		y := op.ArgMax(root.SubScope("input"), op.Const(root.SubScope("input"), test), axis_colum)
		colum_outputlist = append(colum_outputlist, y)
	}

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("ArgMax by row:")
	if output, err := sess.Run(nil, row_outputlist, nil); err != nil {
		panic(err)
	} else {
		for i, j := range output {
			fmt.Println("	", i, " value:", j.Value())
		}
	}
	fmt.Println("ArgMax by colum:")
	if output, err := sess.Run(nil, colum_outputlist, nil); err != nil {
		panic(err)
	} else {
		for i, j := range output {
			fmt.Println("	", i, " value:", j.Value())
		}
	}
}

//测试golang variable
func TestVariable(t *testing.T) {

	// import tensorflow as tf
	// a = tf.Variable(tf.random_normal([2,2],seed=1))
	// b = tf.Variable(tf.truncated_normal([2,2],seed=2))
	// init = tf.global_variables_initializer()
	// with tf.Session() as sess:
	// 	sess.run(init)
	// 	print(sess.run(a))
	// 	print(sess.run(b))
	// #[[-0.8113182   1.4845988 ]  [ 0.06532937 -2.4427042 ]]
	// #[[-0.85811085 -0.19662298]  [ 0.13895045 -1.2212768 ]]

	var (
		root = op.NewScope()
	)

	testdata := []tf.Output{
		op.Const(root.SubScope("input"), []int32{2, 2}),
		op.Const(root.SubScope("input"), [][]int32{{1, 2, 3}, {4, 5, 6}}),
	}
	var vars = make([]*Variable, 0)
	for i, j := range testdata {
		v := NewVariable(root, &j, fmt.Sprintf("init_%d", i))
		vars = append(vars, v)
	}

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("init variables:")
	for _, j := range vars {
		if j != nil {
			j.Init(sess)
		}
	}

	fmt.Println("variable init values:")
	if output, err := sess.Run(nil, testdata, nil); err != nil {
		panic(err)
	} else {
		for i, j := range output {
			fmt.Println("	", i, " value:", j.Value())
		}
	}

	fmt.Println("read variables values:")
	for i, j := range vars {
		if j != nil {
			fmt.Println("	", i, ":", j.Get(sess))
		}
	}
}

//测试tf.TruncatedNormal，返回行或列的最大值下标向量
func TestTruncatedNormal(t *testing.T) {

	// import tensorflow as tf
	// a = tf.Variable(tf.random_normal([2,2],seed=1))
	// b = tf.Variable(tf.truncated_normal([2,2],seed=2))
	// init = tf.global_variables_initializer()
	// with tf.Session() as sess:
	// 	sess.run(init)
	// 	print(sess.run(a))
	// 	print(sess.run(b))
	// #[[-0.8113182   1.4845988 ]  [ 0.06532937 -2.4427042 ]]
	// #[[-0.85811085 -0.19662298]  [ 0.13895045 -1.2212768 ]]

	var (
		root = op.NewScope()
	)

	testdata_withoutseed := []tf.Output{
		op.RandomStandardNormal(root.SubScope("input"),
			op.Const(root.SubScope("input"), []int32{2, 2}),
			tf.Float /* ,	op.RandomStandardNormalSeed(1) */),
		op.RandomStandardNormal(root.SubScope("input"),
			op.Const(root.SubScope("input"), []int32{2, 2}),
			tf.Float),
	}
	testdata_withseed := []tf.Output{
		op.RandomStandardNormal(root.SubScope("input"),
			op.Const(root.SubScope("input"), []int32{2, 2}),
			tf.Float, op.RandomStandardNormalSeed(1)),
		op.RandomStandardNormal(root.SubScope("input"),
			op.Const(root.SubScope("input"), []int32{2, 2}),
			tf.Float, op.RandomStandardNormalSeed(2)),
		op.RandomStandardNormal(root.SubScope("input"),
			op.Const(root.SubScope("input"), []int32{2, 2}),
			tf.Float, op.RandomStandardNormalSeed(3)),
	}

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("RandomStandardNormal values without seeds:")
	if output, err := sess.Run(nil, testdata_withoutseed, nil); err != nil {
		panic(err)
	} else {
		for i, j := range output {
			fmt.Println("	", i, " value:", j.Value())
		}
	}
	fmt.Println("RandomStandardNormal values with seeds:")
	if output, err := sess.Run(nil, testdata_withseed, nil); err != nil {
		panic(err)
	} else {
		for i, j := range output {
			fmt.Println("	", i, " value:", j.Value())
		}
	}
}

func TestVariable_Get(t *testing.T) {

	var (
		root = op.NewScope()
	)

	a := op.Const(root.SubScope("input"), [][]int32{{2, 2}, {2, 2}})
	b := op.Const(root.SubScope("input"), [][]int32{{3, 3}, {3, 3}})

	aa := NewVariable(root, &a, "aa")
	bb := NewVariable(root, &b, "bb")
	fmt.Println(aa.Handle().Op.Name(), bb.Handle().Op.Name())

	A := op.Placeholder(root.SubScope("input"), tf.Int32 /*, op.PlaceholderShape(tf.MakeShape(2, 2))*/)
	B := op.Placeholder(root.SubScope("input"), tf.Int32 /* , op.PlaceholderShape(tf.MakeShape(2, 1)) */)
	product1 := op.MatMul(root, A, B)

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	if aa.Init(sess) && bb.Init(sess) {
		fmt.Println(aa.Get(sess).Value(), bb.Get(sess).Value())
	}

	fmt.Println("values:")
	if output, err := sess.Run(nil, []tf.Output{a, b}, nil); err != nil {
		panic(err)
	} else {
		for i, j := range output {
			fmt.Println("	", i, " value:", j.Value())
		}
	}
	fmt.Println("matual values:")

	if output, err := sess.Run(
		map[tf.Output]*tf.Tensor{
			A: aa.Get(sess),
			B: bb.Get(sess),
		}, []tf.Output{product1}, nil); err != nil {
		panic(err)
	} else {
		for i, j := range output {
			fmt.Println("	", i, " value:", j.Value())
		}
	}

	// type args struct {
	// 	sess *tf.Session
	// }
	// tests := []struct {
	// 	name string
	// 	v    *Variable
	// 	args args
	// 	want tf.Output
	// }{
	// 	// TODO: Add test cases.
	// }
	// for _, tt := range tests {
	// 	t.Run(tt.name, func(t *testing.T) {
	// 		if got := tt.v.Handle(tt.args.sess); !reflect.DeepEqual(got, tt.want) {
	// 			t.Errorf("Variable.Handle() = %v, want %v", got, tt.want)
	// 		}
	// 	})
	// }
}
