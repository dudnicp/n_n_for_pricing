using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{
    class DropoutLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; }

        public double KeepProbability { get; }

        public DropoutLayer(int layerSize, double probability, int batchSize)
        {
            LayerSize = layerSize;
            KeepProbability = probability;
            BatchSize = batchSize;
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
