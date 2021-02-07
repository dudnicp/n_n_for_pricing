using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
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

        public void BackPropagate(ILayer layer, Matrix<double> upstreamWeightedErrors)
        {
            throw new NotImplementedException();
        }

        public void UpdateParameters(ILayer layer)
        {
            throw new NotImplementedException();
        }
    }
}
