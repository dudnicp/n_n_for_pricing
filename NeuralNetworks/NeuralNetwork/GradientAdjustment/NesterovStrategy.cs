using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class NesterovStrategy : IGradientAdjustmentStrategy
    {
        private NesterovParameters Nesterov;

        public Matrix<double> WeightsVelocity { get; set; }
        public Vector<double> BiasVelocity { get; set; }

        public IGradientAdjustmentParameters Parameters => Nesterov;

        public NesterovStrategy(NesterovParameters nesterovParameters)
        {
            Nesterov = nesterovParameters;
        }

        public void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient)
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

        public void Init(int rowCount, int columnCount)
        {
            WeightsVelocity = Matrix<double>.Build.Dense(rowCount, columnCount);
            BiasVelocity = Vector<double>.Build.Dense(columnCount);
        }
    }
}
