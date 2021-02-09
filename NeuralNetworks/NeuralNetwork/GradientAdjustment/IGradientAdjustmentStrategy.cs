using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public interface IGradientAdjustmentStrategy
    {
        IGradientAdjustmentParameters Parameters { get; }
        Matrix<double> WeightsVelocity { get; set; }
        Vector<double> BiasVelocity { get; set; }

        void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient);

        void Init(int rowCount, int columnCount);
    }
}
