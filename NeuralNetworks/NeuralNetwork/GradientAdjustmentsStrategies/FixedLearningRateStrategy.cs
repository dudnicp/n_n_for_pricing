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

        public void BackPropagate(ILayer layer, Matrix<double> upstreamWeightedErrors)
        {
            var standardLayer = layer as BasicStandardLayer;
            // Bias
            upstreamWeightedErrors.PointwiseMultiply(standardLayer.NetInput.Map(standardLayer.Activator.ApplyDerivative), 
                standardLayer.BiasedError);
            standardLayer.BiasedError.Multiply(standardLayer.OnesM, standardLayer.BiasGradient);

            // Weights
            standardLayer.Weights.Multiply(standardLayer.BiasedError, standardLayer.WeightedError);
            standardLayer.PreviousActivation.Multiply(standardLayer.BiasedError.Transpose().Divide(standardLayer.BatchSize), 
                standardLayer.WeightsGradient);

        }

        public void UpdateParameters(ILayer layer)
        {
            var standardLayer = layer as BasicStandardLayer;
            standardLayer.Weights.Subtract(standardLayer.WeightsGradient.Multiply(_parameters.LearningRate), standardLayer.Weights);
            for (int i = 0; i < standardLayer.BatchSize; i++)
            {
                standardLayer.Bias.SetColumn(i, standardLayer.Bias.Column(i).Subtract(
                    standardLayer.BiasGradient.Multiply(_parameters.LearningRate)));
            }
        }
    }
}
