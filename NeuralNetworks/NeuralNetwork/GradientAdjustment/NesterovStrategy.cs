using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class NesterovStrategy : IGradientAdjustmentStrategy
    {
        private NesterovParameters _parameters;
        public IGradientAdjustmentParameters Parameters => _parameters;

        public NesterovStrategy(NesterovParameters parameters)
        {
            _parameters = parameters;
        }

        public void UpdateWeightsAndBiases(BasicStandardLayer layer)
        {
            throw new NotImplementedException();
        }
    }
}
