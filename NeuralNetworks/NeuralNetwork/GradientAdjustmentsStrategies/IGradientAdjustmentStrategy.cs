using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public interface IGradientAdjustmentStrategy
    {
        IGradientAdjustmentParameters Parameters { get; }

        void BackPropagate(ILayer layer, Matrix<double> upstreamWeightedErrors);

        void UpdateParameters(ILayer layer);
    }
}
