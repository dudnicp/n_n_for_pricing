using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class AdamStrategy : IGradientAdjustmentStrategy
    {
        private AdamParameters Adam { get; }
        public IGradientAdjustmentParameters Parameters => Adam;

        private int TrainingStep { get; set; }

        public Matrix<double> WeightsVelocity { get; set; }
        public Vector<double> BiasVelocity { get; set; }

        private Matrix<double> FirstMomentWeights { get; set; }
        private Vector<double> FirstMomentBias { get; set; }
        private Matrix<double> SecondMomentWeights { get; set; }
        private Vector<double> SecondMomentBias { get; set; }
        private Matrix<double> FirstMomentPrimeWeights { get; set; }
        private Vector<double> FirstMomentPrimeBias { get; set; }
        private Matrix<double> SecondMomentPrimeWeights { get; set; }
        private Vector<double> SecondMomentPrimeBias { get; set; }
        private Matrix<double> WeightsVelocityUpdate { get; set; }
        private Vector<double> BiasVelocityUpdate { get; set; }


        public AdamStrategy(AdamParameters adamParameters)
        {
            Adam = adamParameters;
        }

        public void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient)
        {
            if (WeightsVelocity == null)
            {
                Init(weightsGradient.RowCount, weightsGradient.ColumnCount);
            }

            // First Moment
            FirstMomentWeights.Multiply(Adam.FirstMomentDecay, FirstMomentWeights);
            FirstMomentWeights.Add(weightsGradient.Multiply(1 - Adam.FirstMomentDecay), FirstMomentWeights);
            FirstMomentBias.Multiply(Adam.FirstMomentDecay, FirstMomentBias);
            FirstMomentBias.Add(biasGradient.Multiply(1 - Adam.FirstMomentDecay), FirstMomentBias);

            FirstMomentWeights.Divide(1 - Math.Pow(Adam.FirstMomentDecay, TrainingStep), FirstMomentPrimeWeights);
            FirstMomentBias.Divide(1 - Math.Pow(Adam.FirstMomentDecay, TrainingStep), FirstMomentPrimeBias);


            // Second Moment
            SecondMomentWeights.Multiply(Adam.SecondMomentDecay, SecondMomentWeights);
            SecondMomentWeights.Add(weightsGradient.PointwiseMultiply(weightsGradient).
                Multiply(1 - Adam.SecondMomentDecay), SecondMomentWeights);
            SecondMomentBias.Multiply(Adam.SecondMomentDecay, SecondMomentBias);
            SecondMomentBias.Add(biasGradient.PointwiseMultiply(biasGradient).
                Multiply(1 - Adam.SecondMomentDecay), SecondMomentBias);

            SecondMomentWeights.Divide(1 - Math.Pow(Adam.SecondMomentDecay, TrainingStep), SecondMomentPrimeWeights);
            SecondMomentBias.Divide(1 - Math.Pow(Adam.SecondMomentDecay, TrainingStep), SecondMomentPrimeBias);

            // Velocity Update
            SecondMomentPrimeWeights.PointwiseSqrt(SecondMomentPrimeWeights); // r' <- sqrt(r')
            SecondMomentPrimeWeights.Add(Adam.DenominatorFactor); // r' <- r' + delta
            FirstMomentPrimeWeights.PointwiseDivide(SecondMomentPrimeWeights, WeightsVelocityUpdate); // v' <- s' / r'
            WeightsVelocityUpdate.Multiply(Adam.StepSize, WeightsVelocityUpdate); // v' <- eta * v'
            WeightsVelocity.Subtract(WeightsVelocityUpdate); // v <- v - v'

            SecondMomentPrimeBias.PointwiseSqrt(SecondMomentPrimeBias); // r' <- sqrt(r')
            SecondMomentPrimeBias.Add(Adam.DenominatorFactor); // r' <- r' + delta
            FirstMomentPrimeBias.PointwiseDivide(FirstMomentPrimeBias, BiasVelocityUpdate); // v' <- s' / r'
            BiasVelocityUpdate.Multiply(Adam.StepSize, BiasVelocityUpdate); // v' <- eta * v'
            BiasVelocity.Subtract(BiasVelocityUpdate); // v <- v - v'
        }

        public void Init(int rowCount, int columnCount)
        {
            TrainingStep = 1;

            WeightsVelocity = Matrix<double>.Build.Dense(rowCount, columnCount);
            BiasVelocity = Vector<double>.Build.Dense(columnCount);
            FirstMomentWeights = Matrix<double>.Build.Dense(rowCount, columnCount);
            FirstMomentBias = Vector<double>.Build.Dense(columnCount);
            SecondMomentWeights = Matrix<double>.Build.Dense(rowCount, columnCount);
            SecondMomentBias = Vector<double>.Build.Dense(columnCount);
            FirstMomentPrimeWeights = Matrix<double>.Build.Dense(rowCount, columnCount);
            FirstMomentPrimeBias = Vector<double>.Build.Dense(columnCount);
            SecondMomentPrimeWeights = Matrix<double>.Build.Dense(rowCount, columnCount);
            SecondMomentPrimeBias = Vector<double>.Build.Dense(columnCount);
            WeightsVelocityUpdate = Matrix<double>.Build.Dense(rowCount, columnCount);
            BiasVelocityUpdate = Vector<double>.Build.Dense(columnCount);
        }
    }
}
