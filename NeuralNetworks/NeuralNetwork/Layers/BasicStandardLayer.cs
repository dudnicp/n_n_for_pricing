using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;
using NeuralNetwork.Common.Serialization;

namespace NeuralNetwork.Layers
{
    internal class BasicStandardLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> PreviousActivation { get; }

        public Matrix<double> WeightedError { get; }

        public IActivator Activator { get; }

        public Matrix<double> NetInput { get; }

        public Matrix<double> Weights { get; }

        public Matrix<double> Bias { get; }

        public IGradientAdjustmentParameters GradientAdjustmentParameters { get; }

        public Matrix<double> WeightsGradient { get; }

        public Matrix<double> BiasGradient { get; }


        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, 
            IGradientAdjustmentParameters gradientAdjustmentParameters)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Weights = Matrix<double>.Build.DenseOfMatrix(weights);
            Bias = Matrix<double>.Build.DenseOfMatrix(bias);
            NetInput = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            PreviousActivation = Matrix<double>.Build.Dense(InputSize, BatchSize);
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            Activator = activator;
            GradientAdjustmentParameters = gradientAdjustmentParameters;
            WeightedError = Matrix<double>.Build.Dense(InputSize, batchSize);
            WeightsGradient = Matrix<double>.Build.Dense(InputSize, LayerSize);
            BiasGradient = Matrix<double>.Build.Dense(LayerSize, 1);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            upstreamWeightedErrors.PointwiseMultiply(NetInput.Map(Activator.ApplyDerivative), BiasGradient);
            Weights.Multiply(BiasGradient, WeightedError);
            PreviousActivation.Multiply(BiasGradient.Transpose(), WeightsGradient);
        }

        public void Propagate(Matrix<double> input)
        {
            input.CopyTo(PreviousActivation);
            Weights.TransposeThisAndMultiply(input, NetInput);
            NetInput.Add(Bias, NetInput);
            NetInput.Map(Activator.Apply, Activation);
        }

        public void UpdateParameters()
        {
            FixedLearningRateParameters learningRateParameters = GradientAdjustmentParameters as FixedLearningRateParameters;
            Weights.Subtract(WeightsGradient.Multiply(learningRateParameters.LearningRate), Weights);
            Bias.Subtract(BiasGradient.Multiply(learningRateParameters.LearningRate), Bias);
        }
    }
}