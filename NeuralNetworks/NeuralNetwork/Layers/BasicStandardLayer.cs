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

        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            throw new NotImplementedException();
        }


        public void Propagate(Matrix<double> input)
        {
            throw new NotImplementedException();
        }

        public void UpdateParameters()
        {
            throw new NotImplementedException();
        }
    }
}