using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public abstract class AbstractGradientAdjustmentStrategy
    {
        public virtual IGradientAdjustmentParameters Parameters { get; }
        protected Matrix<double> WeightsVelocity { get; set; }
        protected Vector<double> BiasVelocity { get; set; }

        protected abstract void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient);

        public virtual void BackPropagate(BasicStandardLayer layer, Matrix<double> upstreamWeightedError)
        {
            // Bias
            upstreamWeightedError.PointwiseMultiply(layer.NetInput.Map(layer.Activator.ApplyDerivative), layer.BiasedError);
            layer.BiasedError.Multiply(layer.OnesM, layer.BiasGradient);

            // Weights
            layer.Weights.Multiply(layer.BiasedError, layer.WeightedError);
            layer.PreviousActivation.Multiply(layer.BiasedError.Transpose().Divide(layer.BatchSize), layer.WeightsGradient);
        }

        public virtual void UpdateParameters(BasicStandardLayer layer)
        {
            UpdateVelocity(layer.WeightsGradient, layer.BiasGradient);

            layer.Weights.Add(WeightsVelocity, layer.Weights);
            layer.Bias.SetColumn(0, layer.Bias.Column(0).Add(BiasVelocity));
            for (int i = 1; i < layer.BatchSize; i++)
            {
                layer.Bias.SetColumn(i, layer.Bias.Column(0));
            }
        }

        protected virtual void Init(int rowCount, int columnCount)
        {
            WeightsVelocity = Matrix<double>.Build.Dense(rowCount, columnCount);
            BiasVelocity = Vector<double>.Build.Dense(columnCount);
        }
    }
}
