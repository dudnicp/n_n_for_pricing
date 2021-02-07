using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
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
