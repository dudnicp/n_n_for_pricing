using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public class AdamStrategy : AbstractGradientAdjustmentStrategy
    {
        private AdamParameters Adam { get; }
        public override IGradientAdjustmentParameters Parameters => Adam;

        private int TrainingStep { get; set; }

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

        protected override void UpdateVelocity(Matrix<double> weightsGradient, Vector<double> biasGradient)
        {
            if (WeightsVelocity == null)
            {
                Init(weightsGradient.RowCount, weightsGradient.ColumnCount);
            }

            // First Moment
            FirstMomentWeights.Multiply(Adam.FirstMomentDecay, FirstMomentWeights); // s <- rho_1 * s
            FirstMomentWeights.Add(weightsGradient.Multiply(1 - Adam.FirstMomentDecay), FirstMomentWeights); // s <- (1 - rho_1) * g 
            FirstMomentBias.Multiply(Adam.FirstMomentDecay, FirstMomentBias);
            FirstMomentBias.Add(biasGradient.Multiply(1 - Adam.FirstMomentDecay), FirstMomentBias);

            FirstMomentWeights.Divide(1 - Math.Pow(Adam.FirstMomentDecay, TrainingStep), FirstMomentPrimeWeights); // s' <- s / (1 - rho_1^i)
            FirstMomentBias.Divide(1 - Math.Pow(Adam.FirstMomentDecay, TrainingStep), FirstMomentPrimeBias);


            // Second Moment
            SecondMomentWeights.Multiply(Adam.SecondMomentDecay, SecondMomentWeights); // r <- rho_* r
            SecondMomentWeights.Add(weightsGradient.PointwiseMultiply(weightsGradient). 
                Multiply(1 - Adam.SecondMomentDecay), SecondMomentWeights); // r <- r + (1 - rho_2) * g * g
            SecondMomentBias.Multiply(Adam.SecondMomentDecay, SecondMomentBias);
            SecondMomentBias.Add(biasGradient.PointwiseMultiply(biasGradient).
                Multiply(1 - Adam.SecondMomentDecay), SecondMomentBias);

            SecondMomentWeights.Divide(1 - Math.Pow(Adam.SecondMomentDecay, TrainingStep), SecondMomentPrimeWeights);  // r' <- r / (1 - rho_2^i)
            SecondMomentBias.Divide(1 - Math.Pow(Adam.SecondMomentDecay, TrainingStep), SecondMomentPrimeBias);

            // Velocity Update
            SecondMomentPrimeWeights.PointwiseSqrt(SecondMomentPrimeWeights); // r' <- sqrt(r')
            SecondMomentPrimeWeights.Add(Adam.DenominatorFactor); // r' <- r' + delta
            FirstMomentPrimeWeights.PointwiseDivide(SecondMomentPrimeWeights, WeightsVelocityUpdate); // v' <- s' / r'
            WeightsVelocityUpdate.Multiply(-Adam.StepSize, WeightsVelocity); // v' <- eta * v'

            SecondMomentPrimeBias.PointwiseSqrt(SecondMomentPrimeBias); // r' <- sqrt(r')
            SecondMomentPrimeBias.Add(Adam.DenominatorFactor); // r' <- r' + delta
            FirstMomentPrimeBias.PointwiseDivide(SecondMomentPrimeBias, BiasVelocityUpdate); // v' <- s' / r'
            BiasVelocityUpdate.Multiply(-Adam.StepSize, BiasVelocity); // v <- -eta * v'

            TrainingStep++;
        }

        protected override void Init(int rowCount, int columnCount)
        {
            base.Init(rowCount, columnCount);
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
