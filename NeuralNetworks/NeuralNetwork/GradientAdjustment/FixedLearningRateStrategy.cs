using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class FixedLearningRateStrategy : IGradientAdjustmentStrategy
    {
        private FixedLearningRateParameters _parameters;
        public IGradientAdjustmentParameters Parameters => _parameters;

        public FixedLearningRateStrategy(FixedLearningRateParameters parameters)
        {
            _parameters = parameters;
        }

        public void UpdateWeightsAndBiases(BasicStandardLayer layer)
        {
            layer.Weights.Subtract(layer.WeightsGradient.Multiply(_parameters.LearningRate), layer.Weights);
            layer.Bias.SetColumn(0, layer.Bias.Column(0).Subtract(layer.BiasGradient.Multiply(_parameters.LearningRate)));
            for (int i = 1; i < layer.BatchSize; i++)
            {
                layer.Bias.SetColumn(i, layer.Bias.Column(0));
            }
        }
    }
}
