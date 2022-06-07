// A Simple Neural Network with dynamic hidden layers using Gonum ... just for YOU! <3

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/google/uuid"
	"gonum.org/v1/gonum/mat"
)

// as many of you know ... input goes into the neurons ... gets processed ... and output comes out
// here is a flexible neuron size and count
type Net struct {
	id  uuid.UUID
	inp *mat.Dense
	out *mat.Dense
	err *mat.Dense
	neu []*neu   // neurons
	act []string // activation methods
	epo int      // epochs
	rat float64  // learning rate
}

type neu struct {
	inp *mat.Dense
	syn syn
	out *mat.Dense
	err *mat.Dense
	act string // activation method
}

// These w2 and b2 matrices are the temporary storage for updates needed for the main weights and biases
type syn struct {
	w  *mat.Dense
	w2 *mat.Dense
	b  *mat.Dense
	b2 *mat.Dense
}

func main() {

	net := New([]string{"tanh", "tanh", "tanh"}, 10, 0.05, 50, 30, 20, 10)
	i := mat.NewDense(1, 50, nil)
	o := mat.NewDense(1, 10, nil)

	a := net.Forward(i)

	net.Backward([]*mat.Dense{i}, []*mat.Dense{o})

	fmt.Println(a)

}

// comp is the Network composition: first one is the number of inputs and last one is the number of outputs and
// the rest in between are number of synapses in each neuron
// e.g.: for MNIST --> 784, 250, 100, 10
// Remember that there MUST be specified an activation method for each neuron
// Acceptable methods: "tanh", "sigmoid", "relu", "leaky"
func New(activation []string, epo int, rate float64, comp ...int) *Net {

	if len(activation) != len(comp)-1 {
		log.Fatalln("There must be an activation method for every neuron")
	}

	var i, o, e *mat.Dense

	n := make([]*neu, len(comp)-1)

	for a := range comp {

		if a == 0 {
			i = mat.NewDense(1, comp[a], nil)
			o = mat.NewDense(1, comp[len(comp)-1], nil)
			e = mat.NewDense(1, comp[len(comp)-1], nil)
		} else {
			n[a-1] = nneu(activation[a-1], comp[a-1], comp[a])
		}

	}

	return &Net{
		inp: i,
		out: o,
		err: e,
		neu: n,
		act: activation,
		epo: epo,
		rat: rate,
	}

}

// Forward Propagation of the whole Net
func (n *Net) Forward(in ...*mat.Dense) []*mat.Dense {
	outs := make([]*mat.Dense, len(in))

	for a := range in {
		n.inp = in[a]

		n.neu[0].inp = n.inp

		for b := 0; b < len(n.neu); b++ {
			fmt.Println(b)
			if b == len(n.neu)-1 {
				n.out = n.neu[b].forw()
				outs[a] = n.out

			} else {
				n.neu[b+1].inp = n.neu[b].forw()
			}

		}

	}

	return outs

}

// Backward Propagation of the whole Net
func (n *Net) Backward(in, out []*mat.Dense) {

	for a := range in {
		n.inp = in[a]

		for b := 0; b < len(n.neu); b++ {

			if b == 0 {
				n.neu[0].inp = n.inp
				n.neu[1].inp = n.neu[0].forw()

			} else if b == len(n.neu)-1 {
				n.out = n.neu[b].forw()

			} else {
				n.neu[b+1].inp = n.neu[b].forw()
			}

		}

		n.err.Sub(n.out, out[a])
		n.neu[0].err = n.err

		for b := len(n.neu) - 1; b >= 0; b-- {

			if b == 0 {
				n.neu[0].back()
			} else {
				n.neu[b-1].err = n.neu[b].back()
			}
		}

		n.update()

	}
}

// Network weights and biases update
func (n *Net) update() {

	for a := range n.neu {
		n.neu[a].syn.w2.Scale(n.rat, n.neu[a].syn.w2)
		n.neu[a].syn.w.Sub(n.neu[a].syn.w, n.neu[a].syn.w2)

		n.neu[a].syn.b2.Scale(n.rat, n.neu[a].syn.b2)
		n.neu[a].syn.b.Sub(n.neu[a].syn.b, n.neu[a].syn.b2)
	}

}

// New Neuron
func nneu(acitvation string, in, ou int) *neu {
	rand.Seed(time.Now().UnixNano())

	d := make([]float64, in*ou)
	f := make([]float64, ou)

	for a := range d {
		d[a] = rand.NormFloat64()
	}

	for a := range f {
		f[a] = rand.NormFloat64()
	}

	return &neu{
		inp: mat.NewDense(1, in, nil),
		syn: syn{w: mat.NewDense(in, ou, d), w2: mat.NewDense(in, ou, nil), b: mat.NewDense(1, ou, f), b2: mat.NewDense(1, ou, nil)},
		out: mat.NewDense(1, ou, nil),
		err: mat.NewDense(1, ou, nil),
		act: acitvation,
	}

}

// Single Neuron Froward Propagation
func (n *neu) forw() *mat.Dense {
	n.out.Mul(n.inp, n.syn.w)
	n.out.Add(n.out, n.syn.b)

	switch n.act {
	case "tanh":
		n.out.Apply(tanh, n.out)
	case "sigmoid":
		n.out.Apply(sig, n.out)
	case "relu":
		n.out.Apply(relu, n.out)
	default:
		log.Fatalln("bad activation method")
	}

	return n.out

}

// Single Neuron Backward Propagation
func (n *neu) back() *mat.Dense {
	mid := new(mat.Dense)

	switch n.act {
	case "tanh":
		mid.Apply(tanhP, n.out)
	case "sigmoid":
		mid.Apply(sigP, n.out)
	case "relu":
		mid.Apply(reluP, n.out)
	default:
		log.Fatalln("bad activation method")
	}
	mid.MulElem(mid, n.err)

	n.syn.b2 = mat.DenseCopyOf(mid)

	n.syn.w2.Mul(n.inp.T(), mid)

	err := new(mat.Dense)
	err.Mul(mid, n.syn.w.T())

	return err

}

// Activation Methods
func tanh(_, _ int, x float64) float64 { return math.Tanh(x) }

func tanhP(_, _ int, x float64) float64 { return 1.0 - math.Pow(math.Tanh(x), 2) }

func sig(_, _ int, x float64) float64 { return (1.0 / (1 + (math.Pow(math.E, x*(-1.0))))) }

func sigP(_, _ int, x float64) float64 {
	y := (1.0 / (1 + (math.Pow(math.E, x*(-1.0)))))
	return y * (1 - y)
}

func relu(_, _ int, x float64) float64 {
	if x < 0 {
		return 0
	} else {
		return x
	}
}

func reluP(_, _ int, x float64) float64 {
	if x < 0 {
		return 0
	} else {
		return 1
	}
}
