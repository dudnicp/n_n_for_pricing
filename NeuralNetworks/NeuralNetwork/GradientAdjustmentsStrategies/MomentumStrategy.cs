using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class MomentumStrategy : IGradientAdjustmentStrategy
    {
        private MomentumParameters _parameters;
        public IGradientAdjustmentParameters Parameters => _parameters;

        public MomentumStrategy(MomentumParameters parameters)
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
