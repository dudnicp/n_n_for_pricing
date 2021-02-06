using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    internal class BasicStandardLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; }

        public IActivator Activator { get; }

        public Matrix<double> NetInput { get; }

        public Matrix<double> Weights { get; }

        public Matrix<double> Bias { get; }

        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Weights = Matrix<double>.Build.DenseOfMatrix(weights);
            Bias = Matrix<double>.Build.DenseOfMatrix(bias);
            NetInput = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            Activator = activator;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            throw new NotImplementedException();
        }


        public void Propagate(Matrix<double> input)
        {
            Weights.TransposeThisAndMultiply(input).Add(Bias, NetInput);
            NetInput.Map(Activator.Apply, Activation);
        }

        public void UpdateParameters()
        {
            throw new NotImplementedException();
        }
    }
}