using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class AdamStrategy : IGradientAdjustmentStrategy
    {
        private AdamParameters _parameters;
        public IGradientAdjustmentParameters Parameters => _parameters;

        public AdamStrategy(AdamParameters parameters)
        {
            _parameters = parameters;
        }

        public void UpdateWeightsAndBiases(BasicStandardLayer layer)
        {
            throw new NotImplementedException();
        }
    }
}
