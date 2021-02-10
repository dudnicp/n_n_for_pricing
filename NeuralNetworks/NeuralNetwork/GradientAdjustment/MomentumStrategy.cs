using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class MomentumStrategy : AbstractGradientAdjustmentStrategy
    {
        private MomentumParameters Momentum { get; }

        public override IGradientAdjustmentParameters Parameters => Momentum;

        public MomentumStrategy(MomentumParameters momentumParameters)
        {
            Momentum = momentumParameters;
        }

        protected override void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient)
        {
            if (WeightsVelocity == null)
            {
                Init(weightsGradient.RowCount, weightsGradient.ColumnCount);
            }

            // Velocity update
            WeightsVelocity.Multiply(Momentum.Momentum, WeightsVelocity); // v <- v * delta
            WeightsVelocity.Subtract(weightsGradient.Multiply(Momentum.LearningRate), WeightsVelocity); // v <- v - eta * g
            BiasVelocity.Multiply(Momentum.Momentum, BiasVelocity);
            BiasVelocity.Subtract(biasGradient.Multiply(Momentum.LearningRate), BiasVelocity);
        }
    }
}
