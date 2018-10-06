package tfutil

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type Variable struct {
	scope *op.Scope
	dtype tf.DataType

	initVal    *tf.Output
	initHandle *tf.Operation
	readHandle tf.Output

	vHandle *tf.Output

	name string
	// whether has initial
	isInit bool
}

func NewVariable(root *op.Scope, value *tf.Output, name string) (output *Variable) {
	s := root.SubScope("Variable")
	dtype := value.DataType()
	handle := op.VarHandleOp(s, dtype, value.Shape(), op.VarHandleOpSharedName(name))

	init := op.AssignVariableOp(s, handle, *value)
	r := op.ReadVariableOp(s, handle, dtype)

	x := Variable{
		name:    name,
		scope:   s,
		dtype:   dtype,
		initVal: value,
		vHandle: &handle,

		initHandle: init,
		readHandle: r,
	}
	return &x
}

func (v *Variable) Init(sess *tf.Session) bool {
	if v.isInit {
		return true
	}

	if _, err := sess.Run(nil, nil, []*tf.Operation{v.initHandle}); err != nil {
		return false
	} else {
		v.isInit = true
		return true
	}
}

func (v *Variable) Get(sess *tf.Session) interface{} {
	if v.isInit == false {
		return nil
	}

	if out, err := sess.Run(nil, []tf.Output{v.readHandle}, nil); err != nil {
		return nil
	} else {
		return out[0].Value()
	}
}
