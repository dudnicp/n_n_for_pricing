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
        private FixedLearningRateParameters fixedLearningRate;
        public IGradientAdjustmentParameters Parameters => fixedLearningRate;

        public Matrix<double> WeightsVelocity { get; set; }
        public Vector<double> BiasVelocity { get; set; }

        public FixedLearningRateStrategy(FixedLearningRateParameters fixedLearningRateParameters)
        {
            fixedLearningRate = fixedLearningRateParameters;
        }

        public void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient)
        {
            weightsGradient.Multiply(-fixedLearningRate.LearningRate, WeightsVelocity);
            biasGradient.Multiply(-fixedLearningRate.LearningRate, BiasVelocity);
        }

        public void Init(int rowCount, int columnCount)
        {
            throw new NotImplementedException();
        }
    }
}
