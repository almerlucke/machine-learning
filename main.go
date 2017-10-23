package main

import (
	"log"

	"github.com/almerlucke/machine-learning/gradientdescent"

	"gonum.org/v1/gonum/mat"
)

func main() {
	dataPoints := mat.NewDense(4, 3, nil)
	dataPoints.Set(0, 0, 1.0)
	dataPoints.Set(0, 1, 1.0)
	dataPoints.Set(0, 2, 1.0)

	dataPoints.Set(1, 0, 2.0)
	dataPoints.Set(1, 1, 2.0)
	dataPoints.Set(1, 2, 2.0)

	dataPoints.Set(2, 0, 3.0)
	dataPoints.Set(2, 1, 3.0)
	dataPoints.Set(2, 2, 3.0)

	dataPoints.Set(3, 0, 4.0)
	dataPoints.Set(3, 1, 4.0)
	dataPoints.Set(3, 2, 4.0)

	thetas := gradientdescent.GradientDescent(dataPoints, 0.01, gradientdescent.LinearRegressionHypothesis, 1.e-12, nil)
	for i := 0; i < thetas.Len(); i++ {
		log.Printf("theta%d = %f", i, thetas.At(i, 0))
	}
}
