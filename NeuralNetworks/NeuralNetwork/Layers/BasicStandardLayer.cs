using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.GradientAdjustment;
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

        public Matrix<double> BiasedError { get; }

        public IActivator Activator { get; }

        public Matrix<double> NetInput { get; }

        public Matrix<double> Weights { get; }

        public Matrix<double> Bias { get; }

        public Matrix<double> WeightsGradient { get; }

        public Vector<double> BiasGradient { get; }

        public Vector<double> OnesM { get; }

        public IGradientAdjustmentStrategy GradientAdjustmentStrategy { get; set; }


        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, 
            IGradientAdjustmentParameters gradientAdjustmentParameters)
        {
            // General
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;

            // Propagation
            Weights = Matrix<double>.Build.DenseOfMatrix(weights);
            Bias = Matrix<double>.Build.DenseOfMatrix(bias);
            NetInput = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            PreviousActivation = Matrix<double>.Build.Dense(InputSize, BatchSize);
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            Activator = activator;

            // BackPropagation
            GradientAdjustmentStrategy = GradientAdjustmentStrategyFactory.Build(gradientAdjustmentParameters);
            WeightedError = Matrix<double>.Build.Dense(InputSize, BatchSize);
            BiasedError = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            WeightsGradient = Matrix<double>.Build.Dense(InputSize, LayerSize);
            BiasGradient = Vector<double>.Build.Dense(LayerSize);

            // Aux
            OnesM = Vector<double>.Build.Dense(BatchSize, 1.0);
            OnesM.Divide(BatchSize, OnesM);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            GradientAdjustmentStrategy.BackPropagate(this, upstreamWeightedErrors);
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
            GradientAdjustmentStrategy.UpdateParameters(this);
        }
    }
}