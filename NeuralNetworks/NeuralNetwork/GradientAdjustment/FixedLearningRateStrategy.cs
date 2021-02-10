using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class FixedLearningRateStrategy : AbstractGradientAdjustmentStrategy
    {
        private FixedLearningRateParameters fixedLearningRate;
        public override IGradientAdjustmentParameters Parameters => fixedLearningRate;

        public FixedLearningRateStrategy(FixedLearningRateParameters fixedLearningRateParameters)
        {
            fixedLearningRate = fixedLearningRateParameters;
        }

        protected override void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient)
        {
            weightsGradient.Multiply(-fixedLearningRate.LearningRate, WeightsVelocity);
            biasGradient.Multiply(-fixedLearningRate.LearningRate, BiasVelocity);
        }
    }
}
