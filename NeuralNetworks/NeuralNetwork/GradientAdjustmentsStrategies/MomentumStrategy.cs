using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class MomentumStrategy : IGradientAdjustmentStrategy
    {
        private MomentumParameters _parameters;

        private Matrix<double> _weightsVelocity;

        private Vector<double> _biasVelocity;

        public IGradientAdjustmentParameters Parameters => _parameters;

        public MomentumStrategy(MomentumParameters parameters)
        {
            _parameters = parameters;
        }

        public void UpdateWeightsAndBiases(BasicStandardLayer layer)
        {
            // creating velocities if not already created
            if (_weightsVelocity == null)
            {
                _weightsVelocity = Matrix<double>.Build.Dense(layer.InputSize, layer.LayerSize);
                _biasVelocity = Vector<double>.Build.Dense(layer.LayerSize);
            }

            // updating velocities
            _weightsVelocity.Multiply(_parameters.Momentum).Subtract(layer.Weights.Multiply(_parameters.LearningRate), _weightsVelocity);
            _biasVelocity.Multiply(_parameters.Momentum).Subtract(layer.Bias.Column(0).Multiply(_parameters.LearningRate), _biasVelocity);

            // updating parameters
            layer.Weights.Add(_weightsVelocity, layer.Weights);
            layer.Bias.SetColumn(0, layer.Bias.Column(0).Add(_biasVelocity));
            for (int i = 1; i < layer.BatchSize; i++)
            {
                layer.Bias.SetColumn(i, layer.Bias.Column(0));
            }
        }
    }
}
