using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class NesterovStrategy : AbstractGradientAdjustmentStrategy
    {
        private NesterovParameters Nesterov;

        public override IGradientAdjustmentParameters Parameters => Nesterov;

        private Matrix<double> PreprocessedWeights { get; set; }

        public NesterovStrategy(NesterovParameters nesterovParameters)
        {
            Nesterov = nesterovParameters;
        }

        protected override void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient)
        {
            if (WeightsVelocity == null)
            {
                Init(weightsGradient.RowCount, weightsGradient.ColumnCount);
            }

            // Velocity update
            WeightsVelocity.Multiply(Nesterov.Momentum, WeightsVelocity); // v <- v * delta
            WeightsVelocity.Subtract(weightsGradient.Multiply(Nesterov.LearningRate), WeightsVelocity); // v <- v - eta * g
            BiasVelocity.Multiply(Nesterov.Momentum, BiasVelocity);
            BiasVelocity.Subtract(biasGradient.Multiply(Nesterov.LearningRate), BiasVelocity);
        }

        public override void BackPropagate(BasicStandardLayer layer, Matrix<double> upstreamWeightedError)
        {
            layer.Weights.CopyTo(PreprocessedWeights);

            upstreamWeightedError.PointwiseMultiply(layer.NetInput.Map(layer.Activator.ApplyDerivative), layer.BiasedError);
            layer.BiasedError.Multiply(layer.OnesM, layer.BiasGradient);

            // Weights
            PreprocessedWeights.Multiply(layer.BiasedError, layer.WeightedError);
            layer.PreviousActivation.Multiply(layer.BiasedError.Transpose().Divide(layer.BatchSize), layer.WeightsGradient);
        }

        protected override void Init(int rowCount, int columnCount)
        {
            base.Init(rowCount, columnCount);
            PreprocessedWeights = Matrix<double>.Build.Dense(rowCount, columnCount);
        }
    }
}
