package main

import (
	"log"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Hypothesis function type for gradient descent, input is theta vector and feature vector
type Hypothesis func(*mat.VecDense, *mat.VecDense) float64

// Private theta update function
func gradientDescentThetaUpdate(dataPoints *mat.Dense, theta *mat.VecDense, descentRate float64, hypothesis Hypothesis, epsilon float64) bool {
	numRows, numColumns := dataPoints.Dims()

	// Y is added as last element in the feature colummn, so subtract 1 from numColumns to
	// get number of actual features
	numFeatures := numColumns - 1

	// Number of theta is number of features + 1
	numTheta := numColumns

	// Accumulation vector for each element of theta
	accumulations := mat.NewVecDense(numTheta, nil)

	// Multiplier for after accumulation
	multiplier := 1.0 / float64(numRows) * descentRate

	// Accumulate
	for i := 0; i < numRows; i++ {
		// Get feature vector
		features := dataPoints.RowView(i).(*mat.VecDense)

		// Copy features, set feature 0 to 1.0
		x := mat.NewVecDense(numTheta, nil)
		x.SetVec(0, 1.0)
		for f := 0; f < numFeatures; f++ {
			x.SetVec(f+1, features.At(f, 0))
		}

		// Hypothesis
		h := hypothesis(theta, x)

		// Target Y is added as last value in a feature row
		y := x.At(numFeatures, 0)

		// Update accumulations for each theta
		for t := 0; t < numTheta; t++ {
			accumulations.SetVec(t, accumulations.At(t, 0)+(h-y)*x.At(t, 0))
		}
	}

	// Update theta and check totalChange to see if anything is changing still
	totalChange := 0.0

	for t := 0; t < numTheta; t++ {
		change := multiplier * accumulations.At(t, 0)

		// Update change for theta
		theta.SetVec(t, theta.At(t, 0)-change)

		// Update total change
		totalChange += math.Abs(change)
	}

	// Stop if change is smaller than epsilon
	return totalChange < epsilon
}

// GradientDescent calculates best theta values for a matrix of datapoints containing x1...xn, y as row data
// for a given hypothesis. Calculation stops if total change in theta is below epsilon threshold
func GradientDescent(dataPoints *mat.Dense, descentRate float64, hypothesis Hypothesis, epsilon float64, initialTheta *mat.VecDense) *mat.VecDense {
	_, numTheta := dataPoints.Dims()

	// Create theta vector
	theta := mat.NewVecDense(numTheta, nil)

	// Copy initial theta vector if given
	if initialTheta != nil {
		theta.CopyVec(initialTheta)
	}

	stop := false
	for !stop {
		stop = gradientDescentThetaUpdate(dataPoints, theta, descentRate, hypothesis, epsilon)
	}

	return theta
}

// LinearRegressionHypothesis for linear regression
func LinearRegressionHypothesis(theta *mat.VecDense, features *mat.VecDense) float64 {
	var result mat.VecDense

	// Multiply thetas transpose with features
	result.MulVec(theta.T(), features)

	// Get result value from [1, 1] matrix
	return result.At(0, 0)
}

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

	thetas := GradientDescent(dataPoints, 0.01, LinearRegressionHypothesis, 1.e-12, nil)
	for i := 0; i < thetas.Len(); i++ {
		log.Printf("theta%d = %f", i, thetas.At(i, 0))
	}
}
