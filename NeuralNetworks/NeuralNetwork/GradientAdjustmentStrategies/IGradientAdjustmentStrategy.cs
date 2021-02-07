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

        void UpdateWeightsAndBiases(BasicStandardLayer layer);
    }
}
